import sys
import io
import json
import time
import re
import requests
from bs4 import BeautifulSoup
from phi.agent import Agent
from phi.model.groq import Groq
from dotenv import load_dotenv

# Ensure proper UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load environment variables
load_dotenv()

# Initialize the AI agents
labeling_agent = Agent(model=Groq(id="meta-llama/llama-4-maverick-17b-128e-instruct"))
data_agent = Agent(model=Groq(id="meta-llama/llama-4-maverick-17b-128e-instruct"))
checker_agent = Agent(model=Groq(id="meta-llama/llama-4-maverick-17b-128e-instruct"))
clarifier_agent = Agent(model=Groq(id="meta-llama/llama-4-maverick-17b-128e-instruct"))

# Agent roles, goals, and backstories

# Labeler Agent: Analyzes individual content blocks.
LABELER_ROLE = "You are a content analysis expert."
LABELER_GOAL = (
    "Analyze a given HTML content chunk along with its originating HTML tag and determine if "
    "it contains the specific information requested by the user. Return a JSON response indicating "
    "whether it is relevant and, if so, what kind of information is found (e.g., 'movie title', 'rating', 'price')."
)
LABELER_BACKSTORY = (
    "You are trained to precisely understand web content and decide if a particular chunk contains "
    "the information the user is asking for. If the text is clearly a navigational menu (e.g., starts with 'Menu', 'Release Calendar', etc.), "
    "mark it as not relevant. Respond strictly in valid JSON format with no extra commentary."
)

# Scraper Agent: Extracts the answer from a content chunk.
SCRAPER_ROLE = "You are an intelligent information extractor."
SCRAPER_GOAL = (
    "Use the provided label information and HTML context (the selector) to extract precise answers "
    "to the user's query. If the query is open-ended (for example, asking for the movie names), "
    "extract the most likely candidate movie names from the content."
)
SCRAPER_BACKSTORY = "You specialize in analyzing structured web data and retrieving only what's relevant."

# Validator Agent: Validates the extracted answer.
VALIDATOR_ROLE = "You are a data validation agent."
VALIDATOR_GOAL = "Evaluate whether the extracted answer is accurate and aligns with the user's query."
VALIDATOR_BACKSTORY = "You have deep experience validating data pipelines and checking the correctness of AI outputs."

# Clarifier Agent: Provides further analysis when key fields are missing.
CLARIFIER_ROLE = "You are a clarification expert."
CLARIFIER_GOAL = (
    "Analyze the available HTML content and the previous agent analysis. If specific fields such as transaction type are missing, "
    "provide a detailed clarification and suggest follow-up questions to help the user determine the correct value."
)
CLARIFIER_BACKSTORY = (
    "You have a deep understanding of context in structured web data. "
    "When important fields are missing, you offer thoughtful analysis and ask clarifying questions that help retrieve the missing information."
)

def log_message(message):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    sys.stderr.write(f"{timestamp} - {message}\n")
    sys.stderr.flush()

def log_debug_chunk(block, raw_response):
    """Append details about blocks that fail labeling to a debug file."""
    with open("debug_labeling_errors.txt", "a", encoding="utf-8") as f:
        f.write("-----\n")
        f.write(f"Chunk #{block.get('chunk_number')} from <{block.get('selector')}> (Length: {block.get('length')})\n")
        f.write("Chunk Text:\n")
        f.write(block.get("chunk") + "\n")
        f.write("Raw response from labeling agent:\n")
        f.write(raw_response + "\n")
        f.write("-----\n")

def fetch_dynamic_html(url):
    """
    Fetch the webpage using requests and parse it with BeautifulSoup.
    Remove non-visible elements (script, style, meta, noscript, head, link) and return the innerHTML of the body
    (or the whole document if no body tag is found). This approximates the content visible to the user.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "meta", "noscript", "head", "link"]):
            tag.decompose()
        visible_html = soup.body.decode_contents() if soup.body else soup.decode_contents()
        log_message("Visible webpage content loaded successfully!")
        return visible_html
    except Exception as e:
        log_message(f"Error fetching URL {url}: {str(e)}")
        return ""

def extract_semantic_blocks(html):
    """
    Extract semantic blocks from the HTML.
    Each block is a dictionary with:
      - 'chunk_number': sequential number of the block
      - 'chunk': the cleaned text content
      - 'selector': the HTML tag name (e.g., p, div, li)
      - 'length': total number of characters in the chunk
    Save all blocks to a file for debugging.
    """
    soup = BeautifulSoup(html, "html.parser")
    blocks = []
    for idx, tag in enumerate(soup.find_all(['h1', 'h2', 'h3', 'h4', 'strong', 'title', 'p', 'span', 'div', 'li']), start=1):
        text = tag.get_text(strip=True)
        if text and len(text) > 30:
            block = {
                "chunk_number": idx,
                "chunk": text,
                "selector": tag.name,
                "length": len(text)
            }
            blocks.append(block)
    total_chars = sum(block["length"] for block in blocks)
    log_message(f"Total semantic blocks extracted: {len(blocks)} with a combined length of {total_chars} characters.")
    with open("semantic_blocks.txt", "w", encoding="utf-8") as f:
        for block in blocks:
            f.write(json.dumps(block, ensure_ascii=False) + "\n")
    log_message("Saved semantic blocks into semantic_blocks.txt")
    return blocks

def get_keywords(text):
    """
    Extract keywords from text by lowercasing, tokenizing, and removing common stop words.
    """
    stop_words = {"the", "a", "an", "of", "and", "to", "in", "on", "for", "with", "by", "is", "at"}
    words = re.findall(r'\w+', text.lower())
    return [word for word in words if word not in stop_words]

def filter_blocks_generic(blocks, query, threshold=1):
    """
    Generic filtering:
      - Tokenize the query to get significant keywords.
      - For each block, count how many keywords appear.
      - Only keep blocks with an overlap count greater than or equal to threshold.
    """
    keywords = get_keywords(query)
    filtered = []
    for block in blocks:
        block_text = block["chunk"].lower()
        overlap_count = sum(1 for kw in keywords if kw in block_text)
        # Also filter out blocks that are likely navigation if they contain terms like "Menu", "Release Calendar", etc.
        if any(nav in block_text for nav in ["menu", "release calendar", "watchlist", "sign in", "official trailer", "top rated", "features"]):
            continue
        if overlap_count >= threshold:
            block["score"] = overlap_count  # optional metadata
            filtered.append(block)
    log_message(f"Filtered blocks for query '{query}' from {len(blocks)} to {len(filtered)} candidate blocks using threshold {threshold}.")
    return filtered

def label_chunk(block, query):
    """
    Use the labeling agent to determine if the block is relevant to the query.
    Expects a JSON response strictly in this format:
      {"relevant": true, "found": "information type"} or {"relevant": false, "found": "none"}
    """
    chunk = block["chunk"]
    selector = block["selector"]
    prompt = f"""
{LABELER_ROLE}
{LABELER_GOAL}
{LABELER_BACKSTORY}

Determine if the following HTML content chunk is relevant to the user query.
Query: {query}
HTML Selector: <{selector}>
Content:
\"\"\"
{chunk}
\"\"\"

Respond strictly in JSON format as:
    {{"relevant": true, "found": "information type"}}
or
    {{"relevant": false, "found": "none"}}
"""
    try:
        response = labeling_agent.run(prompt)
        if response and hasattr(response, 'content') and response.content:
            content = response.content.strip()
            try:
                result = json.loads(content)
                return result
            except json.JSONDecodeError:
                log_message(f"Labeling JSON decode error. Raw response: {content}")
                log_debug_chunk(block, content)
    except Exception as e:
        log_message(f"Labeling error: {str(e)}")
    return {"relevant": False, "found": "none"}

def extract_data(block, label_info, query):
    """
    Use the data agent to extract an answer from the block.
    The prompt includes the HTML selector for context.
    If the query relates to transaction types (contains 'sale' or 'transaction'),
    instruct the agent to default to 'For Sale' if no explicit type is found.
    """
    chunk = block["chunk"]
    selector = block["selector"]
    fallback_instruction = ""
    if any(keyword in query.lower() for keyword in ["sale", "transaction"]):
        fallback_instruction = "\nIf no explicit transaction type is found in the content, please default to 'For Sale'."
    prompt = f"""
{SCRAPER_ROLE}
{SCRAPER_GOAL}
{SCRAPER_BACKSTORY}

You are given a content chunk with additional context from its HTML tag and the labeling agent.
Query: {query}
Label Information: {json.dumps(label_info)}
HTML Selector: <{selector}>
Content:
\"\"\"
{chunk}
\"\"\"
{fallback_instruction}
Return the response in JSON format as:
    {{"answer": "...", "reason": "..."}}
If not applicable, return:
    {{"answer": "Not available", "reason": "Chunk not relevant."}}
"""
    try:
        response = data_agent.run(prompt)
        if response and hasattr(response, 'content'):
            content = response.content.strip()
            if not content:
                log_message("Extraction error: Received empty response content.")
                return {"answer": "Not available", "reason": "Empty response content."}
            try:
                return json.loads(content)
            except json.JSONDecodeError as jde:
                matches = re.findall(r'\{.*?\}', content, re.DOTALL)
                for match in matches:
                    try:
                        data = json.loads(match)
                        return data
                    except json.JSONDecodeError:
                        continue
                log_message(f"JSON decode error during extraction: {str(jde)}. Full content: {content}")
    except Exception as e:
        log_message(f"Extraction error: {str(e)}")
    return {"answer": "Not available", "reason": "Extraction failed."}

def validate_extraction(block, label_info, query, answer):
    """
    Validate the extracted answer using the checker agent.
    The prompt includes the HTML selector context.
    """
    chunk = block["chunk"]
    selector = block["selector"]
    prompt = f"""
{VALIDATOR_ROLE}
{VALIDATOR_GOAL}
{VALIDATOR_BACKSTORY}

Validate whether the following extracted answer is correct for the query.
Query: {query}
Label Information: {json.dumps(label_info)}
HTML Selector: <{selector}>
Content:
\"\"\"
{chunk}
\"\"\"
Extracted Answer: {answer}

Respond in JSON format as:
    {{"valid": true, "explanation": "..."}} or {{"valid": false, "explanation": "..."}}
"""
    try:
        response = checker_agent.run(prompt)
        if response and hasattr(response, 'content'):
            content = response.content.strip()
            if not content:
                log_message("Checker error: Received empty response content.")
                return {"valid": False, "explanation": "Empty response content."}
            try:
                return json.loads(content)
            except json.JSONDecodeError as jde:
                matches = re.findall(r'\{.*?\}', content, re.DOTALL)
                for match in matches:
                    try:
                        validation = json.loads(match)
                        return validation
                    except json.JSONDecodeError:
                        continue
                log_message(f"Checker JSON decode error: {str(jde)}. Full content: {content}")
    except Exception as e:
        log_message(f"Checker error: {str(e)}")
    return {"valid": False, "explanation": "Validation failed."}

def run_agents_sequentially(blocks, query):
    """
    Process candidate blocks:
      1. Label each block for relevance.
      2. If relevant, extract the answer.
      3. Validate the extraction.
    Stop on the first block that passes validation.
    """
    for block in blocks:
        log_message(f"Processing block #{block['chunk_number']} from <{block['selector']}> (length {block['length']} characters).")
        label_info = label_chunk(block, query)
        log_message(f"Labeling result for block #{block['chunk_number']}: {json.dumps(label_info)}")
        if not label_info.get("relevant", False):
            log_message(f"Block #{block['chunk_number']} deemed not relevant. Skipping.")
            continue
        extraction = extract_data(block, label_info, query)
        answer = extraction.get("answer", "Not available")
        if isinstance(answer, list):
            answer_str = ", ".join(str(item) for item in answer)
        elif isinstance(answer, str):
            answer_str = answer
        else:
            answer_str = str(answer)
        log_message(f"Extracted answer for block #{block['chunk_number']}: {answer_str}")
        if answer_str.strip().lower() != "not available":
            check = validate_extraction(block, label_info, query, answer_str)
            if check.get("valid"):
                log_message(f"A valid answer has been found in block #{block['chunk_number']}. Stopping further processing.")
                return {
                    "label_info": label_info,
                    "answer": answer,
                    "reason": extraction.get("reason", ""),
                    "validation": check.get("explanation", "No explanation provided.")
                }
            else:
                log_message(f"Block #{block['chunk_number']} answer failed validation.")
    log_message("No valid answer was found after processing all candidate blocks.")
    return {
        "label_info": None,
        "answer": "Not available",
        "reason": "No relevant block found.",
        "validation": "No valid answer passed the checker."
    }

def clarify_missing_field(blocks, query):
    """
    Use the clarifier agent to analyze the aggregated HTML content and previous analysis
    when key fields (e.g., transaction type) are missing.
    Returns a clarification and further analysis to help the user form follow-up questions.
    """
    # Aggregate text from all blocks and limit to avoid excessive length.
    aggregated_text = " ".join([block["chunk"] for block in blocks])
    if len(aggregated_text) > 2000:
        aggregated_text = aggregated_text[:2000] + "..."
    prompt = f"""
{CLARIFIER_ROLE}
{CLARIFIER_GOAL}
{CLARIFIER_BACKSTORY}

Query: {query}
Aggregated Content:
\"\"\"
{aggregated_text}
\"\"\"

Based on the aggregated content above and the prior agent analysis, please provide a detailed clarification regarding the missing fields (for example, transaction type) and suggest any follow-up questions the user might consider to narrow down the correct value.
Return the response in JSON format as:
    {{"clarification": "...", "analysis": "..."}}.
"""
    try:
        response = clarifier_agent.run(prompt)
        if response and hasattr(response, 'content'):
            content = response.content.strip()
            try:
                return json.loads(content)
            except json.JSONDecodeError as jde:
                log_message(f"Clarifier agent JSON decode error: {str(jde)}. Content: {content}")
    except Exception as e:
        log_message(f"Clarifier agent error: {str(e)}")
    return {"clarification": "Not available", "analysis": "Clarification failed."}

def main():
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Missing URL or Query parameter"}))
        sys.exit(1)
    url = sys.argv[1]
    query = sys.argv[2]
    html = fetch_dynamic_html(url)
    blocks = extract_semantic_blocks(html)
    log_message(f"Extracted {len(blocks)} semantic blocks from the webpage.")
    
    # Step 1: Filter blocks for query
    candidate_blocks = filter_blocks_generic(blocks, query, threshold=1)
    
    # Step 2: If filtering yields no candidates, fallback to processing all blocks.
    if len(candidate_blocks) == 0:
        log_message(f"No candidate blocks after filtering for query: '{query}'. Running agents on all blocks instead.")
        candidate_blocks = blocks
    
    # Step 3: Run agents sequentially on the candidate blocks.
    result = run_agents_sequentially(candidate_blocks, query)
    
    # Step 4: If key field (like transaction type) is not found, invoke the clarifier agent.
    if result["answer"].strip().lower() == "not available" and any(keyword in query.lower() for keyword in ["sale", "transaction"]):
        log_message("Missing field detected. Invoking clarifier agent for additional analysis.")
        clarifier_result = clarify_missing_field(blocks, query)
        result["clarification"] = clarifier_result.get("clarification", "Not available")
        result["analysis"] = clarifier_result.get("analysis", "No analysis provided.")
    
    output = {
        "url": url,
        "query": query,
        "label_info": result.get("label_info"),
        "answer": result.get("answer"),
        "reason": result.get("reason"),
        "validation": result.get("validation"),
        "clarification": result.get("clarification", ""),
        "analysis": result.get("analysis", "")
    }
    print(json.dumps(output, indent=2))
    sys.stdout.flush()
    sys.exit(0)

if __name__ == "__main__":
    main()
