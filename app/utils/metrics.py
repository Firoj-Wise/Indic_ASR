import jiwer
import re
import string
from app.utils.logger_utils import LOGGER
from app.constants import log_msg

def normalize_text(text: str) -> str:
    """
    Normalizes text for ASR evaluation.
    - Removes punctuation (Standard + Danda).
    - Removes all whitespace (for split-word robustness).
    
    Args:
        text (str): Input text.
        
    Returns:
        str: Normalized text.
    """
    if not text:
        LOGGER.warning(log_msg.WARN_EMPTY_TEXT_NORM)
        return ""
    
    # Define punctuation to remove: standard + Indic Dandas + common separators
    # We avoid [^\w\s] regex because it can strip Indic vowel signs (Matras).
    punctuation = string.punctuation + "।" + "॥" + "–" + "—"
    
    # Remove punctuation using translate
    translator = str.maketrans('', '', punctuation)
    text = text.translate(translator)
    
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
