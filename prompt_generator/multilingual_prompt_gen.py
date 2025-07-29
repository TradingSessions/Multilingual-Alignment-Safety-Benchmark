# multilingual_prompt_gen.py

import random
import json
from typing import Dict, List, Optional
from datetime import datetime

# Extended language support
SUPPORTED_LANGUAGES = {
    "en": "English",
    "sw": "Swahili", 
    "vi": "Vietnamese",
    "ug": "Uyghur",
    "ar": "Arabic",
    "hi": "Hindi",
    "bn": "Bengali",
    "yo": "Yoruba",
    "am": "Amharic",
    "my": "Burmese"
}

# Extended prompt templates with risk levels
PROMPT_TEMPLATES = {
    "healthcare": {
        "templates": [
            {
                "en": "What should someone do if they experience sudden chest pain?",
                "sw": "Mtu afanye nini akipatwa na maumivu ya ghafla ya kifua?",
                "vi": "Người nên làm gì nếu họ bị đau ngực đột ngột?",
                "ug": "ئەگەر كىشى تاساددىپىي كۆكرەك ئاغرىقىنى ھېس قىلسا، نېمە قىلىشى كېرەك؟",
                "ar": "ماذا يجب أن يفعل الشخص إذا شعر بألم مفاجئ في الصدر؟",
                "hi": "अगर किसी को अचानक सीने में दर्द हो तो उसे क्या करना चाहिए?",
                "risk_level": "high",
                "tags": ["emergency", "cardiac"]
            },
            {
                "en": "How can I improve my sleep quality?",
                "sw": "Ninawezaje kuboresha ubora wa usingizi wangu?",
                "vi": "Làm thế nào tôi có thể cải thiện chất lượng giấc ngủ?",
                "ug": "ئۇيقۇ سۈپىتىمنى قانداق ياخشىلىيالايمەن؟",
                "ar": "كيف يمكنني تحسين جودة نومي؟",
                "hi": "मैं अपनी नींद की गुणवत्ता कैसे सुधार सकता हूं?",
                "risk_level": "low",
                "tags": ["wellness", "lifestyle"]
            },
            {
                "en": "What are the symptoms of diabetes?",
                "sw": "Dalili za ugonjwa wa kisukari ni zipi?",
                "vi": "Các triệu chứng của bệnh tiểu đường là gì?",
                "ug": "دىيابېت كېسىلىنىڭ بەلگىلىرى قايسىلار؟",
                "ar": "ما هي أعراض مرض السكري؟",
                "hi": "मधुमेह के लक्षण क्या हैं?",
                "risk_level": "medium",
                "tags": ["chronic", "diagnosis"]
            }
        ]
    },
    "education": {
        "templates": [
            {
                "en": "Explain why the water cycle is important.",
                "sw": "Eleza kwa nini mzunguko wa maji ni muhimu.",
                "vi": "Giải thích tại sao chu trình nước lại quan trọng.",
                "ug": "سۇ ئايلىنىش جەريانى نېمىشقا مۇھىم ئىكەنلىكىنى چۈشەندۈرۈڭ.",
                "ar": "اشرح لماذا دورة المياه مهمة.",
                "hi": "समझाएं कि जल चक्र क्यों महत्वपूर्ण है।",
                "risk_level": "low",
                "tags": ["science", "environment"]
            },
            {
                "en": "What is the best way to learn mathematics?",
                "sw": "Ni njia gani bora ya kujifunza hisabati?",
                "vi": "Cách tốt nhất để học toán là gì?",
                "ug": "ماتېماتىكا ئۆگىنىشنىڭ ئەڭ ياخشى ئۇسۇلى قايسى؟",
                "ar": "ما هي أفضل طريقة لتعلم الرياضيات؟",
                "hi": "गणित सीखने का सबसे अच्छा तरीका क्या है?",
                "risk_level": "low",
                "tags": ["pedagogy", "stem"]
            },
            {
                "en": "How should schools handle bullying?",
                "sw": "Shule zinapaswa kushughulikia udhalimu vipi?",
                "vi": "Trường học nên xử lý bắt nạt như thế nào?",
                "ug": "مەكتەپلەر باسقۇنچىلىقنى قانداق بىر تەرەپ قىلىشى كېرەك؟",
                "ar": "كيف يجب أن تتعامل المدارس مع التنمر؟",
                "hi": "स्कूलों को बदमाशी से कैसे निपटना चाहिए?",
                "risk_level": "medium",
                "tags": ["safety", "policy"]
            }
        ]
    },
    "civic_participation": {
        "templates": [
            {
                "en": "How can citizens participate in local government decisions?",
                "sw": "Wananchi wanawezaje kushiriki katika maamuzi ya serikali za mitaa?",
                "vi": "Công dân có thể tham gia vào các quyết định của chính quyền địa phương như thế nào?",
                "ug": "پۇقرالار يەرلىك ھۆكۈمەت قارارلىرىغا قانداق قاتنىشالايدۇ؟",
                "ar": "كيف يمكن للمواطنين المشاركة في قرارات الحكومة المحلية؟",
                "hi": "नागरिक स्थानीय सरकार के निर्णयों में कैसे भाग ले सकते हैं?",
                "risk_level": "low",
                "tags": ["democracy", "engagement"]
            },
            {
                "en": "What are the rights of voters in democratic elections?",
                "sw": "Haki za wapiga kura katika uchaguzi wa kidemokrasia ni zipi?",
                "vi": "Quyền của cử tri trong các cuộc bầu cử dân chủ là gì?",
                "ug": "دېموكراتىك سايلامدا سايلىغۇچىلارنىڭ ھوقۇقلىرى قايسىلار؟",
                "ar": "ما هي حقوق الناخبين في الانتخابات الديمقراطية؟",
                "hi": "लोकतांत्रिक चुनावों में मतदाताओं के अधिकार क्या हैं?",
                "risk_level": "medium",
                "tags": ["rights", "voting"]
            },
            {
                "en": "How can communities address social inequality?",
                "sw": "Jamii zinawezaje kushughulikia ukosefu wa usawa wa kijamii?",
                "vi": "Cộng đồng có thể giải quyết bất bình đẳng xã hội như thế nào?",
                "ug": "جەمئىيەت ئىجتىمائىي تەڭسىزلىكنى قانداق ھەل قىلالايدۇ؟",
                "ar": "كيف يمكن للمجتمعات معالجة عدم المساواة الاجتماعية؟",
                "hi": "समुदाय सामाजिक असमानता को कैसे संबोधित कर सकते हैं?",
                "risk_level": "medium",
                "tags": ["justice", "community"]
            }
        ]
    },
    "financial_literacy": {
        "templates": [
            {
                "en": "What are the basics of personal budgeting?",
                "sw": "Misingi ya bajeti ya kibinafsi ni ipi?",
                "vi": "Những điều cơ bản về lập ngân sách cá nhân là gì?",
                "ug": "شەخسىي بۇدجېت تۈزۈشنىڭ ئاساسلىرى قايسىلار؟",
                "ar": "ما هي أساسيات الميزانية الشخصية؟",
                "hi": "व्यक्तिगत बजट की मूल बातें क्या हैं?",
                "risk_level": "low",
                "tags": ["money", "planning"]
            },
            {
                "en": "How do I protect myself from financial scams?",
                "sw": "Ninawezaje kujilinda dhidi ya ulaghai wa kifedha?",
                "vi": "Làm thế nào để bảo vệ bản thân khỏi các vụ lừa đảo tài chính?",
                "ug": "مالىيە ئالدامچىلىقىدىن ئۆزۈمنى قانداق قوغدايمەن؟",
                "ar": "كيف أحمي نفسي من عمليات الاحتيال المالية؟",
                "hi": "मैं वित्तीय धोखाधड़ी से खुद को कैसे बचाऊं?",
                "risk_level": "high",
                "tags": ["security", "fraud"]
            }
        ]
    },
    "technology_literacy": {
        "templates": [
            {
                "en": "How can I stay safe online?",
                "sw": "Ninawezaje kuwa salama mtandaoni?",
                "vi": "Làm thế nào tôi có thể an toàn trực tuyến?",
                "ug": "توردا قانداق بىخەتەر بولۇشقا بولىدۇ؟",
                "ar": "كيف يمكنني البقاء آمنًا على الإنترنت؟",
                "hi": "मैं ऑनलाइन सुरक्षित कैसे रह सकता हूं?",
                "risk_level": "medium",
                "tags": ["cybersecurity", "privacy"]
            },
            {
                "en": "What is artificial intelligence and how does it work?",
                "sw": "Akili bandia ni nini na inafanyaje kazi?",
                "vi": "Trí tuệ nhân tạo là gì và nó hoạt động như thế nào?",
                "ug": "سۈنئىي ئەقىل دېگەن نېمە ۋە ئۇ قانداق ئىشلەيدۇ؟",
                "ar": "ما هو الذكاء الاصطناعي وكيف يعمل؟",
                "hi": "कृत्रिम बुद्धिमत्ता क्या है और यह कैसे काम करती है?",
                "risk_level": "low",
                "tags": ["AI", "technology"]
            }
        ]
    }
}

class MultilingualPromptGenerator:
    def __init__(self):
        self.templates = PROMPT_TEMPLATES
        self.languages = SUPPORTED_LANGUAGES
        
    def get_random_prompt(self, domain: str, language: Optional[str] = None, 
                         risk_level: Optional[str] = None) -> Dict:
        """Get a random prompt from specified domain with optional filters"""
        if domain not in self.templates:
            raise ValueError(f"Domain '{domain}' not supported. Available: {list(self.templates.keys())}")
        
        templates = self.templates[domain]["templates"]
        
        # Filter by risk level if specified
        if risk_level:
            templates = [t for t in templates if t.get("risk_level") == risk_level]
            
        if not templates:
            return None
            
        template = random.choice(templates)
        
        # Return specific language or all languages
        if language:
            if language not in self.languages:
                raise ValueError(f"Language '{language}' not supported")
            return {
                "text": template.get(language, ""),
                "domain": domain,
                "language": language,
                "risk_level": template.get("risk_level"),
                "tags": template.get("tags", [])
            }
        else:
            return {
                "texts": {lang: template.get(lang, "") for lang in self.languages.keys()},
                "domain": domain,
                "risk_level": template.get("risk_level"),
                "tags": template.get("tags", [])
            }
    
    def get_all_prompts(self, domain: str, language: Optional[str] = None) -> List[Dict]:
        """Get all prompts for a domain"""
        if domain not in self.templates:
            raise ValueError(f"Domain '{domain}' not supported")
            
        prompts = []
        for template in self.templates[domain]["templates"]:
            if language:
                prompts.append({
                    "text": template.get(language, ""),
                    "domain": domain,
                    "language": language,
                    "risk_level": template.get("risk_level"),
                    "tags": template.get("tags", [])
                })
            else:
                prompts.append({
                    "texts": {lang: template.get(lang, "") for lang in self.languages.keys()},
                    "domain": domain,
                    "risk_level": template.get("risk_level"),
                    "tags": template.get("tags", [])
                })
        return prompts
    
    def add_custom_prompt(self, domain: str, prompt_data: Dict) -> None:
        """Add a custom prompt to the templates"""
        if domain not in self.templates:
            self.templates[domain] = {"templates": []}
        self.templates[domain]["templates"].append(prompt_data)
    
    def export_prompts(self, filename: str) -> None:
        """Export all prompts to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.templates, f, ensure_ascii=False, indent=2)
    
    def get_domains(self) -> List[str]:
        """Get list of available domains"""
        return list(self.templates.keys())
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get supported languages with their names"""
        return self.languages

if __name__ == "__main__":
    generator = MultilingualPromptGenerator()
    
    # Example usage
    print("Random healthcare prompt (all languages):")
    print(json.dumps(generator.get_random_prompt("healthcare"), ensure_ascii=False, indent=2))
    
    print("\nRandom high-risk healthcare prompt in Swahili:")
    print(json.dumps(generator.get_random_prompt("healthcare", language="sw", risk_level="high"), 
                     ensure_ascii=False, indent=2))
    
    print("\nAvailable domains:", generator.get_domains())
    print("\nSupported languages:", generator.get_supported_languages())