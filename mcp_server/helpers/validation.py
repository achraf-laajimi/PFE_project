def is_valid_student_id(student_id: str) -> bool:
    """
    Returns True if the ID is a valid resolved string.
    Returns False if it's a placeholder alias or empty.
    """
    if not student_id:
        return False
    
    # List of illegal aliases that should have been resolved by the planner
    invalid_aliases = ("CURRENT_STUDENT", "UNKNOWN", "")
    
    if student_id.strip().upper() in invalid_aliases:
        return False
        
    return True