"""
ClinVar Agent - Queries ClinVar API for variant information and associated conditions
"""

import requests
import json
import re
import os
from typing import Dict, Any, List
from groq import Groq


class ClinVarAgent:
    """Agent responsible for querying ClinVar API to find variant information and associated conditions"""

    def __init__(self):
        """
        Initialize the ClinVar agent
        """
        self.api_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        self.esummary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        self.headers = {
            "Accept-Encoding": "gzip"
        }
        
        # Initialize Groq LLM for doctor-friendly formatting
        try:
            groq_api_key = os.getenv("GROQ_API_KEY")
            if groq_api_key:
                self.groq_client = Groq(api_key=groq_api_key)
                self.llm_model = "qwen/qwen3-32b"
                print("✅ Groq LLM initialized for ClinVar formatting")
            else:
                self.groq_client = None
                print("⚠️ GROQ_API_KEY not found, ClinVar formatting will be disabled")
        except Exception as e:
            print(f"⚠️ Failed to initialize Groq LLM: {e}")
            self.groq_client = None

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Query ClinVar API for variant information and associated conditions

        Args:
            state: Current workflow state containing 'parsed_data'

        Returns:
            Updated state with 'clinvar_results'
        """
        try:
            parsed_data = state.get("parsed_data", {})
            gene = parsed_data.get("gene", "NA")
            variant = parsed_data.get("variant", "NA")

            # Check if we have valid data
            if gene == "NA" and variant == "NA":
                return {**state, "clinvar_results": []}

            # Query ClinVar API and get raw data
            raw_clinvar_data = self._query_clinvar(gene, variant)
            
            # Generate doctor-friendly report for EACH variant entry
            # LLM will identify epilepsy-related syndromes
            doctor_reports = []  # List of {variant_id, title, report, syndromes}
            all_epilepsy_syndromes = []  # Collect syndromes from all variants
            
            if raw_clinvar_data and gene != "NA" and variant != "NA":
                for variant_id, variant_data in raw_clinvar_data.items():
                    # Skip metadata entries
                    if variant_id == "uids" or not isinstance(variant_data, dict):
                        continue
                    
                    # Generate report for this specific variant
                    # LLM returns both report and epilepsy syndromes
                    single_variant_data = {variant_id: variant_data}
                    report, epilepsy_syndromes = self._format_clinvar_for_doctors(single_variant_data, gene, variant)
                    
                    # Get title for display
                    title = variant_data.get("title", f"Variant {variant_id}")
                    
                    doctor_reports.append({
                        "variant_id": variant_id,
                        "title": title,
                        "report": report,
                        "syndromes": epilepsy_syndromes  # Syndromes from LLM for this variant
                    })
                    
                    # Collect all syndromes
                    all_epilepsy_syndromes.extend(epilepsy_syndromes)
            
            # Remove duplicates from syndromes across all variants
            clinvar_syndromes = list(set(all_epilepsy_syndromes))
            
            # Print the ClinVar results
            print("\n" + "="*60)
            print("🧬 CLINVAR AGENT RESULTS")
            print("="*60)
            print(f"Gene queried: {gene}")
            print(f"Variant queried: {variant}")
            print(f"ClinVar entries found: {len(doctor_reports)}")
            print(f"Doctor reports generated: {len(doctor_reports)}")
            print(f"Epilepsy syndromes identified by LLM: {clinvar_syndromes}")
            print("="*60)
            
            return {
                **state, 
                "clinvar_syndromes": clinvar_syndromes,
                "clinvar_doctor_reports": doctor_reports,
                "clinvar_raw": raw_clinvar_data
            }

        except Exception as e:
            print(f"ClinVar API error: {e}")
            raise e

    def _format_clinvar_for_doctors(self, raw_clinvar_data: Dict[str, Any], gene: str, variant: str) -> tuple[str, List[str]]:
        """
        Format raw ClinVar response into doctor-friendly format using Groq LLM
        
        Args:
            raw_clinvar_data: Raw ClinVar API response data
            gene: Gene symbol being queried
            variant: Variant notation being queried
            
        Returns:
            Tuple of (doctor_friendly_report, epilepsy_syndromes_list)
        """
        if not self.groq_client:
            raise ValueError("GROQ_API_KEY not available. LLM is required for ClinVar formatting.")
        
        try:
            # Create an epilepsy-focused prompt for the LLM
            # Pass raw JSON directly - no serialization needed
            raw_json_str = json.dumps(raw_clinvar_data, indent=2)
            
            prompt = f"""You are an expert in genetic epilepsy and epilepsy genetics. 

GENE: {gene}
VARIANT: {variant}

RAW CLINVAR DATA (JSON):
{raw_json_str}

INSTRUCTIONS:
Generate a clinical report for epileptologists with the following sections (extract from the JSON provided):

1. **Variant Summary**
   - Extract from: title, obj_type, variation_set[0].cdna_change, protein_change
   - Include: Gene symbol, variant notation (protein and cDNA changes), variant type
   - Chromosomal location from: variation_set[0].variation_loc (use 'current' status)
   
2. **Clinical Significance**
   - Extract from: germline_classification.description, germline_classification.review_status, germline_classification.last_evaluated
   - Include pathogenicity classification, review status, last evaluation date
   - Provide brief clinical interpretation for epileptologists
   
3. **Epilepsy Syndromes**
   - Extract from: germline_classification.trait_set
   - For each epilepsy-related syndrome in trait_set:
     * Get trait_name
     * Extract sources from trait_xrefs (look for OMIM, Orphanet, MONDO)
     * Format as: "Syndrome Name (OMIM:######, Orphanet:####)"
   - Focus on syndromes containing: epilepsy, seizure, EIEE, Dravet, Lennox, West syndrome, etc.
   
4. **Clinical Phenotypes (HPO)**
   - Extract from: germline_classification.trait_set
   - For traits with trait_xrefs containing "Human Phenotype Ontology":
     * Format as: "Phenotype name (HPO:HP:#######)"
   - Emphasize epilepsy-related phenotypes
   
5. **Other Associated Conditions**
   - List non-epilepsy conditions from trait_set with their database sources
   - Skip generic terms like "Inborn genetic diseases" or "not provided"
   - Include MedGen, MeSH, and other relevant database references

6. **Molecular Details** (if available)
   - Extract from: molecular_consequence_list, protein_change
   - Include cross-references from: variation_set[0].variation_xrefs (dbSNP, ClinGen)

NOTE: The JSON structure may vary. Extract available information and keep language clear for epileptologists.

After the clinical report, provide a JSON list of ONLY epilepsy-related syndromes on a new line:

EPILEPSY_SYNDROMES_JSON
["Syndrome 1", "Syndrome 2", "Syndrome 3"]

Rules for the syndrome list:
• Include ONLY syndromes related to: epilepsy, seizure, EIEE, Dravet, Lennox, West syndrome, convulsion
• EXCLUDE: Generic terms ("Inborn genetic diseases"), HPO phenotypes alone, "not provided"
• If no epilepsy syndromes found: []

Example format:

EPILEPSY_SYNDROMES_JSON
["Developmental and epileptic encephalopathy", "Benign familial neonatal seizures, 1"]
"""
            # Call Groq LLM
            response = self.groq_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in epilepsy genetics who helps epileptologists and neurologists interpret genetic variant data for patients with epilepsy. You provide clear, structured clinical summaries that emphasize epilepsy phenotypes, seizure characteristics, and developmental outcomes."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            formatted_result = response.choices[0].message.content
            
            # Debug: Check if thinking tags are present
            if "<think>" in formatted_result:
                print("🔍 Found <think> tags in LLM response, removing them...")
                original_length = len(formatted_result)
            
            # Remove thinking tags if present - do this BEFORE any other processing
            if "<think>" in formatted_result and "</think>" in formatted_result:
                # Remove everything between <think> and </think> including the tags
                formatted_result = re.sub(r'<think>.*?</think>', '', formatted_result, flags=re.DOTALL)
                formatted_result = formatted_result.strip()
            
            # Also remove any standalone thinking tags that might be present
            formatted_result = re.sub(r'<think>.*?</think>', '', formatted_result, flags=re.DOTALL | re.IGNORECASE)
            formatted_result = formatted_result.strip()
            
            # Debug: Show what was removed
            if "<think>" in response.choices[0].message.content:
                new_length = len(formatted_result)
                print(f"   Removed {original_length - new_length} characters of thinking content")
            
            # Extract the doctor-friendly report and epilepsy syndromes
            doctor_report, epilepsy_syndromes = self._parse_llm_response(formatted_result)
            
            print("✅ ClinVar data formatted using Groq LLM")
            print(f"   Epilepsy syndromes identified by LLM: {epilepsy_syndromes}")
            
            return doctor_report, epilepsy_syndromes
            
        except Exception as e:
            print(f"❌ LLM formatting failed: {e}")
            raise

    def _parse_llm_response(self, llm_output: str) -> tuple[str, List[str]]:
        """
        Parse LLM response to extract doctor-friendly report and epilepsy syndromes.
        Expects the marker "EPILEPSY_SYNDROMES_JSON" followed by a JSON array.
        """
        try:
            # First, remove any remaining thinking tags that might have been missed
            llm_output = re.sub(r'<think>.*?</think>', '', llm_output, flags=re.DOTALL | re.IGNORECASE)
            
            # Clean up any unwanted headers that might appear
            # Remove various forms of OUTPUT headers
            llm_output = re.sub(r'^\s*\*?\*?OUTPUT\s+1\s*-?\s*CLINICAL\s+REPORT\*?\*?\s*:?\s*\n?', '', llm_output, flags=re.IGNORECASE | re.MULTILINE).strip()
            
            # Look for the EPILEPSY_SYNDROMES_JSON marker
            marker = "EPILEPSY_SYNDROMES_JSON"
            search_start = llm_output.find(marker)
            
            if search_start != -1:
                # Everything before the marker is the clinical report
                report_part = llm_output[:search_start].strip()
                # Everything from the marker onwards is where we search for the JSON
                search_region = llm_output[search_start:]
            else:
                # No marker found, look for old-style markers as fallback
                old_markers = ["OUTPUT 2 - EPILEPSY SYNDROMES", "EPILEPSY SYNDROMES", "**EPILEPSY SYNDROMES**"]
                for old_marker in old_markers:
                    old_search_start = llm_output.find(old_marker)
                    if old_search_start != -1:
                        report_part = llm_output[:old_search_start].strip()
                        search_region = llm_output[old_search_start:]
                        break
                else:
                    # No markers found at all
                    report_part = llm_output.strip()
                    search_region = llm_output

            # JSON array pattern (supports escaped quotes and code fences)
            # First, try to extract from ```json ... ``` blocks
            json_fence_pattern = r'```(?:json)?\s*(\[(?:\s*"(?:[^\\"\n]|\\.)*"\s*(?:,\s*"(?:[^\\"\n]|\\.)*"\s*)*)?\])\s*```'
            fence_match = re.search(json_fence_pattern, search_region, flags=re.DOTALL)
            
            if fence_match:
                array_text = fence_match.group(1)
            else:
                # No code fence, look for plain JSON array
                json_array_pattern = r'\[(?:\s*"(?:[^\\"\n]|\\.)*"\s*(?:,\s*"(?:[^\\"\n]|\\.)*"\s*)*)?\]'
                array_match = re.search(json_array_pattern, search_region, flags=re.DOTALL)
                array_text = array_match.group(0) if array_match else None
            
            if array_text:
                try:
                    epilepsy_syndromes = json.loads(array_text)
                    if isinstance(epilepsy_syndromes, list):
                        return report_part, epilepsy_syndromes
                except json.JSONDecodeError:
                    pass

            # If we get here, no valid JSON list found
            return report_part, []

        except Exception:
            return llm_output.strip(), []

    def _query_clinvar(self, gene: str, variant: str) -> Dict[str, Any]:
        """
        Query ClinVar API for variant information and associated conditions
        
        Args:
            gene: Gene symbol
            variant: Variant notation
            
        Returns:
            Raw ClinVar data dictionary
        """
        try:
            # Build search query with gene and variant
            search_terms = []
            
            # Add gene if available
            if gene != "NA":
                search_terms.append(f"{gene}[gene]")
            
            # Add variant if available
            if variant != "NA":
                search_terms.append(f'"{variant}"')
            
            # Combine search terms
            search_query = " AND ".join(search_terms)
            
            print(f"ClinVar search query: {search_query}")

            # Search for variant IDs and get raw data
            raw_data = self._search_clinvar_ids(search_query)
            
            return raw_data

        except requests.RequestException as e:
            print(f"ClinVar API request failed: {e}")
            return {}

    def _search_clinvar_ids(self, search_query: str) -> Dict[str, Any]:
        """
        Search for ClinVar variant IDs and get detailed information
        
        Args:
            search_query: Search query string
            
        Returns:
            Raw ClinVar data dictionary
        """
        try:
            search_params = {
                "db": "clinvar",
                "term": search_query,
                "retmax": 20,
                "retmode": "json"
            }

            search_response = requests.get(self.api_url, params=search_params, headers=self.headers)

            if search_response.status_code == 200:
                search_data = search_response.json()
                variant_ids = search_data.get("esearchresult", {}).get("idlist", [])
                
                if not variant_ids:
                    print("No ClinVar entries found for the search query")
                    return {}

                print(f"Found {len(variant_ids)} ClinVar entries")

                # Get detailed information for each variant
                raw_data = self._get_variant_details(variant_ids)
                return raw_data
            else:
                print(f"ClinVar search failed with status {search_response.status_code}")
                return {}

        except Exception as e:
            print(f"Error in ClinVar search: {e}")
            return {}

    def _get_variant_details(self, variant_ids: List[str]) -> Dict[str, Any]:
        """
        Get detailed information for ClinVar variants
        
        Args:
            variant_ids: List of ClinVar variant IDs
            
        Returns:
            Raw ClinVar data dictionary
        """
        try:
            # Get detailed summaries
            summary_params = {
                "db": "clinvar",
                "id": ",".join(variant_ids),
                "retmode": "json"
            }

            summary_response = requests.get(self.esummary_url, params=summary_params, headers=self.headers)

            if summary_response.status_code == 200:
                summary_data = summary_response.json()
                results = summary_data.get("result", {})
                return results
            else:
                print(f"ClinVar summary request failed with status {summary_response.status_code}")
                return {}

        except Exception as e:
            print(f"Error getting variant details: {e}")
            return {}


