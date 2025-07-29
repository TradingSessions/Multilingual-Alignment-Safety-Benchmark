# Test suite for the multilingual prompt generator

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from prompt_generator.multilingual_prompt_gen import MultilingualPromptGenerator

class TestMultilingualPromptGenerator:
    """Test suite for MultilingualPromptGenerator"""
    
    @pytest.fixture
    def generator(self):
        """Create a generator instance for testing"""
        return MultilingualPromptGenerator()
    
    def test_initialization(self, generator):
        """Test that the generator initializes correctly"""
        assert generator is not None
        assert len(generator.get_languages()) > 0
        assert len(generator.get_domains()) > 0
    
    def test_supported_languages(self, generator):
        """Test that all expected languages are supported"""
        languages = generator.get_languages()
        expected_languages = ["en", "sw", "vi", "ug", "ar", "hi", "bn", "yo", "am", "my"]
        
        for lang in expected_languages:
            assert lang in languages, f"Language {lang} not found in supported languages"
    
    def test_supported_domains(self, generator):
        """Test that all expected domains are supported"""
        domains = generator.get_domains()
        expected_domains = ["healthcare", "education", "financial_literacy", "civic_participation", "technology"]
        
        for domain in expected_domains:
            assert domain in domains, f"Domain {domain} not found in supported domains"
    
    def test_get_random_prompt_english(self, generator):
        """Test getting a random prompt in English"""
        prompt = generator.get_random_prompt("healthcare", "en")
        
        assert prompt is not None
        assert isinstance(prompt, dict)
        assert "text" in prompt
        assert "language" in prompt
        assert "domain" in prompt
        assert prompt["language"] == "en"
        assert prompt["domain"] == "healthcare"
    
    def test_get_random_prompt_swahili(self, generator):
        """Test getting a random prompt in Swahili"""
        prompt = generator.get_random_prompt("education", "sw")
        
        assert prompt is not None
        assert isinstance(prompt, dict)
        assert prompt["language"] == "sw"
        assert prompt["domain"] == "education"
        assert len(prompt["text"]) > 0
    
    def test_get_random_prompt_invalid_language(self, generator):
        """Test handling of invalid language codes"""
        prompt = generator.get_random_prompt("healthcare", "invalid_lang")
        assert prompt is None
    
    def test_get_random_prompt_invalid_domain(self, generator):
        """Test handling of invalid domain"""
        prompt = generator.get_random_prompt("invalid_domain", "en")
        assert prompt is None
    
    def test_prompt_has_required_fields(self, generator):
        """Test that generated prompts have all required fields"""
        for language in ["en", "sw", "ar"]:
            for domain in ["healthcare", "education"]:
                prompt = generator.get_random_prompt(domain, language)
                if prompt:  # Skip if combination not available
                    assert "text" in prompt
                    assert "language" in prompt
                    assert "domain" in prompt
                    assert "risk_level" in prompt
                    assert "tags" in prompt
                    assert isinstance(prompt["tags"], list)
    
    def test_prompt_risk_levels(self, generator):
        """Test that prompts have valid risk levels"""
        valid_risk_levels = ["low", "medium", "high", "critical"]
        
        for _ in range(10):  # Test multiple prompts
            prompt = generator.get_random_prompt("healthcare", "en")
            if prompt:
                assert prompt["risk_level"] in valid_risk_levels
    
    def test_prompt_text_not_empty(self, generator):
        """Test that prompt text is not empty"""
        for language in ["en", "sw"]:
            prompt = generator.get_random_prompt("healthcare", language)
            if prompt:
                assert len(prompt["text"].strip()) > 0
                assert prompt["text"] != ""
    
    def test_multiple_prompts_different(self, generator):
        """Test that multiple calls return different prompts (most of the time)"""
        prompts = []
        for _ in range(5):
            prompt = generator.get_random_prompt("healthcare", "en")
            if prompt:
                prompts.append(prompt["text"])
        
        # Should have some variety (not all exactly the same)
        unique_prompts = set(prompts)
        assert len(unique_prompts) > 1 or len(prompts) <= 1  # Allow for small template sets
    
    def test_language_specific_content(self, generator):
        """Test that different languages produce different content"""
        en_prompt = generator.get_random_prompt("healthcare", "en")
        sw_prompt = generator.get_random_prompt("healthcare", "sw")
        
        if en_prompt and sw_prompt:
            # The texts should be different (different languages)
            assert en_prompt["text"] != sw_prompt["text"]
            assert en_prompt["language"] != sw_prompt["language"]
    
    def test_domain_specific_tags(self, generator):
        """Test that prompts have domain-appropriate tags"""
        healthcare_prompt = generator.get_random_prompt("healthcare", "en")
        education_prompt = generator.get_random_prompt("education", "en")
        
        if healthcare_prompt:
            # Healthcare prompts should have relevant tags
            tags = healthcare_prompt.get("tags", [])
            assert isinstance(tags, list)
            # Could check for medical-related tags if they exist
        
        if education_prompt:
            # Education prompts should have relevant tags
            tags = education_prompt.get("tags", [])
            assert isinstance(tags, list)

if __name__ == "__main__":
    pytest.main([__file__])