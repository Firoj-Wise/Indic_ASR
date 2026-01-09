from typing import List, Dict, Tuple

def add_text_to_diarization_segments(
    word_timestamps: List[Tuple[str, float, float]],  # [(word, start, end), ...]
    diarization: List[Dict]                           # [{"speaker": ..., "start": ..., "end": ...}, ...]
) -> List[Dict]:
    """
    Adds transcribed text to each diarization segment based on word timestamps.
    
    For each diarization segment, finds all words that fall within its time range
    and adds them as the 'text' field.
    
    Returns:
        List of diarization segments with 'text' field added.
    """
    if not diarization:
        return []
    
    result = []
    
    for seg in diarization:
        seg_start = seg["start"]
        seg_end = seg["end"]
        
        # Find all words whose midpoint falls within this segment
        segment_words = []
        for word, w_start, w_end in word_timestamps:
            word_mid = (w_start + w_end) / 2
            if seg_start <= word_mid <= seg_end:
                segment_words.append(word)
        
        result.append({
            "speaker": seg["speaker"],
            "start": seg["start"],
            "end": seg["end"],
            "text": " ".join(segment_words)
        })
    
    return result