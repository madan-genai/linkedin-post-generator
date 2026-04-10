import json
from llm_helper import llm
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

import re

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    text = text.encode("utf-8", "ignore").decode("utf-8")
    
    # remove invalid surrogate characters
    text = re.sub(r'[\ud800-\udfff]', '', text)

    return text

def process_posts(raw_file_path, processed_file_path=None):
    with open(raw_file_path, encoding='utf-8') as file:
        posts = json.load(file)
        enriched_posts = []
        for post in posts:
            clean_post = clean_text(post["text"])
            metadata = extract_metadata(clean_post)
            post_with_metadata = post | metadata
            enriched_posts.append(post_with_metadata)

    unified_tags = get_unified_tags(enriched_posts)
    for post in enriched_posts:
        current_tags = post['tags']
        new_tags = {unified_tags.get(tag, tag) for tag in current_tags}
        post['tags'] = list(new_tags)

    with open(processed_file_path, encoding='utf-8', mode="w") as outfile:
        json.dump(enriched_posts, outfile, indent=4)

def extract_metadata(post):
    template = """
You are an information extraction system.

Your task is to analyze a LinkedIn post and extract structured metadata.

INSTRUCTIONS:
1. Return ONLY a valid JSON object.
2. Do NOT include explanations, text, or formatting outside the JSON.
3. The JSON must contain EXACTLY these three keys:
   - "line_count"
   - "language"
   - "tags"

RULES:

Line Count:
• Count the total number of lines in the post.
• A line is defined as text separated by a newline character.

Language:
• Detect the primary language used in the post.
• Allowed values:
  - "English"
  - "Urdu"
  - "Urdu + English"

Tags:
• Extract up to TWO relevant tags from the post.
• Tags should NOT include the "#" symbol.
• Tags should represent the main topics of the post.
• If hashtags exist, prioritize them.

OUTPUT FORMAT (STRICT):
{{
  "line_count": integer,
  "language": "English | Urdu | Urdu + English",
  "tags": ["tag1", "tag2"]
}}

Analyze the following LinkedIn post:

{post}
"""
    
    pt=PromptTemplate.from_template(template)
    # defining the chaining system in langchain
    chain = pt | llm
    # chain then pass to invoke function to generate answer
    response = chain.invoke(input={"post":post})
    # trying Json parser if possible or OutparserException if not possible to parse
    try:
        Json_parser=JsonOutputParser()
        res = Json_parser.parse(response.content)
    except OutputParserException:
        raise OutputParserException("Content too big. Unable to parse the jobs.")
    return res


def get_unified_tags(posts_with_metadata):
    unique_tags = set()
    # Loop through each post and extract the tags
    for post in posts_with_metadata:
        unique_tags.update(post['tags'])  # Add the tags to the set

    unique_tags_list = ','.join(unique_tags)


    template = '''I will give you a list of tags. You need to unify tags with the following requirements,
         1. Tags are unified and merged to create a shorter list. 
         Example 1: "Jobseekers", "Job Hunting" can be all merged into a single tag "Job Search". 
         Example 2: "Motivation", "Inspiration", "Drive" can be mapped to "Motivation"
         Example 3: "Personal Growth", "Personal Development", "Self Improvement" can be mapped to "Self Improvement"
         Example 4: "Scam Alert", "Job Scam" etc. can be mapped to "Scams"
         2. Each tag should be follow title case convention. example: "Motivation", "Job Search"
         3. Output should be a JSON object, No preamble
         3. Output should have mapping of original tag and the unified tag. 
         For example: {{"Jobseekers": "Job Search",  "Job Hunting": "Job Search", "Motivation": "Motivation}}
    
    Here is the list of tags: 
    {tags}
    '''

    pt = PromptTemplate.from_template(template)

    chain = pt | llm
    response=chain.invoke(input={"tags":str( unique_tags_list)})
    try:
        json_parser = JsonOutputParser()
        res=json_parser.parse(response.content)
    except OutputParserException:
        raise OutputParserException("Content too big. Unable to parse the jobs.")
    return res


if __name__ == "__main__":
    process_posts("data/raw_posts.json", "data/processed_posts.json")