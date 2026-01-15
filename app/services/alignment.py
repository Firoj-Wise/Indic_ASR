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

    # BETTER APPROACH: Single pass based on words to ensure no text is lost (orphaned).
    # Logic: Iterate words, assign each to the "best" segment (closest).
    
    # Initialize buckets for each segment
    segment_buckets = [[] for _ in range(len(diarization))]
    
    for w in normalized_words:
        word_mid = (w["start"] + w["end"]) / 2
        
        best_seg_idx = -1
        min_dist = float('inf')
        
        # Check all segments
        for i, seg in enumerate(diarization):
            start, end = seg["start"], seg["end"]
            
            if start <= word_mid <= end:
                # Perfect match (containment) 
                dist = 0
            else:
                # Distance to boundary
                dist = min(abs(word_mid - start), abs(word_mid - end))
            
            # Update best
            if dist < min_dist:
                min_dist = dist
                best_seg_idx = i
            elif dist == min_dist and dist == 0:
                pass
        
        # Assign word to best bucket
        # We assign it even if distance is large, because preserving text is priority over silence.
        if best_seg_idx != -1:
            segment_buckets[best_seg_idx].append(w["word"])
            
    # Build final result list
    result = []
    for i, seg in enumerate(diarization):
        text = " ".join(segment_buckets[i])
        
        result.append({
            "speaker": seg["speaker"],
            "start": seg["start"],
            "end": seg["end"],
            "text": text
        })

    return result