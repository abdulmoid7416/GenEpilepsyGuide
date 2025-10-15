"""
Input Parser Agent - Extracts structured data from patient descriptions
"""

import json
from typing import Dict, Any
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


class InputParserAgent:
    """Agent responsible for parsing patient descriptions into structured data"""
    
    def __init__(self, groq_api_key: str):
        """Initialize the input parser agent"""
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model="qwen/qwen3-32b",
            temperature=0
        )
        self.prompt = ChatPromptTemplate.from_template("""
        Parse the following patient description into a structured dictionary. Extract:
        - gene: Gene name (e.g., 'TSC2', 'SCN1A')
        - variant: Variant notation (e.g., 'p.Arg905Gln', 'c.1234G>A')
        - variant_type: Type of variant (missense, nonsense, frameshift, etc.)
        - demographics: Dictionary with age, sex, ethnicity if mentioned
        - phenotypes: List of symptoms/features mentioned

        Use 'NA' for any missing fields. Return ONLY a valid JSON dictionary, with no other text or formatting.

        Patient description: {input}
        """)
        # Create a chain using the newer syntax
        self.chain = self.prompt | self.llm
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse patient description into structured data
        
        Args:
            state: Current workflow state containing 'input'
            
        Returns:
            Updated state with 'parsed_data'
        """
        try:
            patient_input = state["input"]
            
            # Run the LLM chain to parse the input
            result = self.chain.invoke({"input": patient_input})
            result = result.content
            
            # Check if we got an empty response
            if not result or result.strip() == "":
                raise ValueError("Empty response from LLM")
            
            # Clean and parse the JSON response
            parsed_data = self._clean_and_parse_json(result)
            
            # Print the parsed results
            print("\n" + "="*60)
            print("📋 INPUT PARSER RESULTS")
            print("="*60)
            print(f"Gene: {parsed_data.get('gene', 'N/A')}")
            print(f"Variant: {parsed_data.get('variant', 'N/A')}")
            print(f"Variant Type: {parsed_data.get('variant_type', 'N/A')}")
            print(f"Demographics: {parsed_data.get('demographics', {})}")
            print(f"Phenotypes: {parsed_data.get('phenotypes', [])}")
            print("="*60)
            
            return {
                **state,
                "parsed_data": parsed_data
            }
            
        except Exception as e:
            print(f"Error parsing input: {e}")
            return {
                **state,
                "parsed_data": {
                    "gene": "NA",
                    "variant": "NA", 
                    "variant_type": "NA",
                    "demographics": {},
                    "phenotypes": []
                }
            }
    
    def _clean_and_parse_json(self, result: str) -> Dict[str, Any]:
        """
        Clean LLM response and parse JSON
        
        Args:
            result: Raw LLM response
            
        Returns:
            Parsed dictionary
        """
        # Clean the response
        result = result.strip()
        print(f"Cleaned response: {repr(result)}")
        
        # Remove thinking tags if present
        if "<think>" in result and "</think>" in result:
            result = result.split("</think>")[-1].strip()
        
        # Remove code blocks if present
        if result.startswith("```json"):
            result = result[7:-3]
        elif result.startswith("```"):
            result = result[3:-3]
        
        
        try:
            # Parse JSON
            parsed = json.loads(result)
            print(f"Successfully parsed JSON: {parsed}")
            return parsed
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Attempting to parse: {result}")
            
            # Try to find JSON-like content in the response
            import re
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                print(f"Found JSON-like content: {repr(json_str)}")
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
            
            # If all parsing fails, return a default structure
            print("Failed to parse JSON, returning default structure")
            return {
                "gene": "NA",
                "variant": "NA", 
                "variant_type": "NA",
                "demographics": {},
                "phenotypes": []
            }
