from typing import List, Dict, Tuple, Any
from app.utils.logger_utils import LOGGER

def add_text_to_diarization_segments(
    word_timestamps: List[Any],  # List of tuples or dicts
    diarization: List[Dict]      # [{"speaker": ..., "start": ..., "end": ...}, ...]
) -> List[Dict]:
    """
    Adds transcribed text to each diarization segment based on word timestamps.
    """
    if not diarization:
        return []
        
    if not word_timestamps:
        LOGGER.warning("No word timestamps provided for alignment")
        return diarization
    
    # Normalize word_timestamps to list of dicts for easier handling
    normalized_words = []
    for item in word_timestamps:
        word, start, end = None, None, None
        
        if isinstance(item, dict):
            word = item.get('word') or item.get('text')
            start = item.get('start')
            end = item.get('end')
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            word = item[0]
            start = item[1]
            end = item[2]
            
        if word and start is not None and end is not None:
             normalized_words.append({"word": word, "start": float(start), "end": float(end)})
             
    if not normalized_words:
        LOGGER.warning("No valid timestamps found after normalization (all None?)")
        
    result = []
    
    for seg in diarization:
        seg_start = seg["start"]
        seg_end = seg["end"]
        
        # Find all words whose midpoint falls within this segment
        segment_words = []
        for w in normalized_words:
            word_mid = (w["start"] + w["end"]) / 2
            if seg_start <= word_mid <= seg_end:
                segment_words.append(w["word"])
        
        result.append({
            "speaker": seg["speaker"],
            "start": seg["start"],
            "end": seg["end"],
            "text": " ".join(segment_words)
        })
    
    return result