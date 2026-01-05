import jiwer
import re
from app.utils.logger_utils import LOGGER
from app.constants import log_msg

def normalize_text(text: str) -> str:
    """
    Normalizes text for ASR evaluation.
    - Removes punctuation (mostly).
    - Collapses multiple spaces.
    - Strips leading/trailing whitespace.
    
    Args:
        text (str): Input text.
        
    Returns:
        str: Normalized text.
    """
    if not text:
        LOGGER.warning(log_msg.WARN_EMPTY_TEXT_NORM)
        return ""
    
    # Remove common punctuation but keep intra-word characters if needed
    # For Indic languages, we might want to be careful.
    # A simple regex for basic punctuation removal:
    # This regex removes characters that are NOT alphanumeric and NOT whitespace.
    # Note: \w in Python regex matches Unicode word characters (including Devanagari).
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove all whitespace to handle split-word issues (e.g., 'परि पाठी' -> 'परिपाठी')
    text = re.sub(r'\s+', '', text)
    
    return text

def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    Calculates Character Error Rate (CER).
    
    Args:
        reference (str): Ground truth text.
        hypothesis (str): ROI (Recognized Output from Inference).
        
    Returns:
        float: CER value (0.0 to 1.0+).
    """
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)
    
    if not ref_norm:
        # If reference is empty, any hypothesis is 100% error unless empty too
        return 1.0 if hyp_norm else 0.0
        
    return jiwer.cer(ref_norm, hyp_norm)

def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Calculates Word Error Rate (WER).
    """
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)
    
    if not ref_norm:
        return 1.0 if hyp_norm else 0.0

    return jiwer.wer(ref_norm, hyp_norm)
