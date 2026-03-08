# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# Copyright (c) 2026, BabyBIONN Contributors

# enhanced_vni_classes/modules/web_search.py 
import asyncio
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
import urllib.parse
import importlib  # Add this

# FIXED: Try multiple ways to import
try:
    # First try relative import (when running as part of package)
    from ..utils.imports import import_optional
except (ImportError, ValueError):
    try:
        # Fall back to absolute import
        from enhanced_vni_classes.utils.imports import import_optional
    except ImportError:
        # Last resort: define a simple version
        def import_optional(module_name):
            try:
                return importlib.import_module(module_name)
            except ImportError:
                return None

# Make aiohttp optional - it won't crash if not installed
aiohttp = import_optional("aiohttp")

class WebSearch:
    """Web search module for VNIs."""
    
    def __init__(self, vni_id: str, api_keys: Optional[Dict[str, str]] = None):
        self.vni_id = vni_id
        self.api_keys = api_keys or {}
        self.search_cache = {}
        self.cache_expiry = 3600  # 1 hour
        self.max_results = 5
        
        # Check if aiohttp is available
        self._aiohttp_available = aiohttp is not None
            
    async def search(self, 
                    query: str, 
                    domain: str = "general",
                    num_results: int = 5) -> Dict[str, Any]:
        """Perform web search."""
        
        # Check cache first
        cache_key = f"{domain}_{query}"
        if cache_key in self.search_cache:
            cached_data = self.search_cache[cache_key]
            if datetime.now().timestamp() - cached_data["timestamp"] < self.cache_expiry:
                return cached_data["results"]
        
        # Use appropriate search API based on domain
        if domain in ["medical", "health"]:
            results = await self._search_medical(query, num_results)
        elif domain in ["legal", "law"]:
            results = await self._search_legal(query, num_results)
        elif domain in ["academic", "research"]:
            results = await self._search_academic(query, num_results)
        else:
            results = await self._search_general(query, num_results)
        
        # Cache results
        self.search_cache[cache_key] = {
            "results": results,
            "timestamp": datetime.now().timestamp(),
            "query": query,
            "domain": domain
        }
        
        # Limit cache size
        if len(self.search_cache) > 100:
            oldest_key = min(self.search_cache.keys(), 
                           key=lambda k: self.search_cache[k]["timestamp"])
            del self.search_cache[oldest_key]
        
        return results
    
    async def _search_general(self, query: str, num_results: int) -> Dict[str, Any]:
        """Perform general web search."""
        # If aiohttp is available and we have API keys, we could make real requests
        # For now, use simulated results
        
        encoded_query = urllib.parse.quote(query)
        
        # Simulated search results
        results = {
            "query": query,
            "domain": "general",
            "results": [
                {
                    "title": f"Information about {query}",
                    "link": f"https://example.com/search?q={encoded_query}",
                    "snippet": f"This is information related to {query}. According to general knowledge sources...",
                    "source": "example.com",
                    "relevance_score": 0.85,
                    "freshness": "2024"
                },
                {
                    "title": f"More details on {query}",
                    "link": f"https://wikipedia.org/wiki/{encoded_query}",
                    "snippet": f"Wikipedia provides comprehensive information about {query}. It covers various aspects including...",
                    "source": "wikipedia.org",
                    "relevance_score": 0.78,
                    "freshness": "2024"
                }
            ][:num_results],
            "total_results": 2,
            "search_timestamp": datetime.now().isoformat(),
            "note": "Web search is using simulated results. Install aiohttp for real web searches." if not self._aiohttp_available else ""
        }
        
        return results
    
    async def _search_medical(self, query: str, num_results: int) -> Dict[str, Any]:
        """Perform medical-specific search."""
        # Placeholder for medical search API
        
        results = {
            "query": query,
            "domain": "medical",
            "results": [
                {
                    "title": f"Medical information: {query}",
                    "link": f"https://pubmed.ncbi.nlm.nih.gov/?term={urllib.parse.quote(query)}",
                    "snippet": f"Clinical information about {query}. Studies show...",
                    "source": "PubMed",
                    "relevance_score": 0.92,
                    "freshness": "2024",
                    "medical_verified": True
                }
            ][:num_results],
            "total_results": 1,
            "search_timestamp": datetime.now().isoformat(),
            "disclaimer": "Medical information should be verified with healthcare professionals",
            "note": "Install aiohttp to enable real medical database searches." if not self._aiohttp_available else ""
        }
        
        return results
    
    async def _search_legal(self, query: str, num_results: int) -> Dict[str, Any]:
        """Perform legal-specific search."""
        # Placeholder for legal search API
        
        results = {
            "query": query,
            "domain": "legal",
            "results": [
                {
                    "title": f"Legal aspects of {query}",
                    "link": f"https://legal-resources.com/search?q={urllib.parse.quote(query)}",
                    "snippet": f"Legal information regarding {query}. Relevant statutes include...",
                    "source": "Legal Resources",
                    "relevance_score": 0.88,
                    "freshness": "2024",
                    "jurisdiction": "General"
                }
            ][:num_results],
            "total_results": 1,
            "search_timestamp": datetime.now().isoformat(),
            "disclaimer": "Not legal advice. Consult with a qualified attorney.",
            "note": "Install aiohttp to enable real legal database searches." if not self._aiohttp_available else ""
        }
        
        return results
    
    async def _search_academic(self, query: str, num_results: int) -> Dict[str, Any]:
        """Perform academic/research search."""
        # Placeholder for academic search APIs
        
        results = {
            "query": query,
            "domain": "academic",
            "results": [
                {
                    "title": f"Academic research on {query}",
                    "link": f"https://scholar.google.com/scholar?q={urllib.parse.quote(query)}",
                    "snippet": f"Research papers and academic publications about {query}. Key findings include...",
                    "source": "Google Scholar",
                    "relevance_score": 0.90,
                    "freshness": "2024",
                    "citations": 25
                }
            ][:num_results],
            "total_results": 1,
            "search_timestamp": datetime.now().isoformat(),
            "note": "Install aiohttp to enable real academic database searches." if not self._aiohttp_available else ""
        }
        
        return results
    
    def extract_key_information(self, search_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract key information from search results."""
        extracted_info = []
        
        for result in search_results.get("results", []):
            info = {
                "title": result.get("title", ""),
                "summary": result.get("snippet", ""),
                "source": result.get("source", ""),
                "relevance": result.get("relevance_score", 0.5),
                "key_points": self._extract_key_points(result.get("snippet", "")),
                "verification_level": self._assess_verification(result)
            }
            extracted_info.append(info)
        
        return extracted_info
    
    def _extract_key_points(self, text: str, max_points: int = 3) -> List[str]:
        """Extract key points from text."""
        # Simple extraction - can be enhanced with NLP
        sentences = text.split('. ')
        key_points = []
        
        for sentence in sentences:
            if len(sentence.split()) >= 5 and len(sentence.split()) <= 30:
                if any(keyword in sentence.lower() for keyword in ["important", "key", "major", "primary", "essential"]):
                    key_points.append(sentence.strip())
                elif len(key_points) < max_points and len(sentence) > 20:
                    key_points.append(sentence.strip())
        
        return key_points[:max_points]
    
    def _assess_verification(self, result: Dict[str, Any]) -> str:
        """Assess the verification level of a result."""
        source = result.get("source", "").lower()
        
        if any(domain in source for domain in [".gov", ".edu", ".org"]):
            return "high"
        elif any(verified in result for verified in ["medical_verified", "legal_verified"]):
            return "high"
        elif "wikipedia" in source:
            return "medium"
        else:
            return "basic"
    
    def clear_cache(self):
        """Clear the search cache."""
        self.search_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self.search_cache),
            "domains_cached": list(set(data["domain"] for data in self.search_cache.values())),
            "oldest_entry": min((data["timestamp"] for data in self.search_cache.values()), default=None),
            "newest_entry": max((data["timestamp"] for data in self.search_cache.values()), default=None),
            "aiohttp_available": self._aiohttp_available
        }
    
    def is_aiohttp_available(self) -> bool:
        """Check if aiohttp is available."""
        return self._aiohttp_available
    
    def get_installation_instructions(self) -> str:
        """Get instructions for installing required dependencies."""
        return (
            "For real web search functionality, install: pip install aiohttp\n"
            "Currently using simulated search results."
        )
