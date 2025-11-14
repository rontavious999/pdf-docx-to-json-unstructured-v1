"""
Text preprocessing functions for line cleanup, normalization, and soft-wrap coalescing.

This module contains functions for:
- Normalizing Unicode glyphs to ASCII representations
- Collapsing spaced-out letters and capitals
- Reading text files with encoding detection
- Detecting section headings and category headers
- Scrubbing headers/footers and practice information
- Coalescing soft-wrapped lines
"""

import re
from pathlib import Path
from typing import List, Set
from collections import Counter

# Import constants from the constants module
from .constants import (
    CHECKBOX_ANY, BULLET_RE, PAGE_NUM_RE, ADDRESS_LIKE_RE,
    DENTAL_PRACTICE_EMAIL_RE, BUSINESS_WITH_ADDRESS_RE,
    PRACTICE_NAME_PATTERN, KNOWN_FIELD_LABELS
)

# Import OCR correction functions (Category 2 Fix 2.2)
from .ocr_correction import (
    preprocess_text_with_ocr_correction,
    preprocess_field_label,
    restore_ligatures
)


def normalize_glyphs_line(s: str) -> str:
    """
    Normalize Unicode checkbox and bullet glyphs to ASCII representations.
    
    Converts various Unicode symbols for checkboxes, bullets, and checkmarks
    into standardized ASCII patterns like "[ ]", "[x]", and "•".
    
    Enhanced with OCR correction (Category 2 Fix 2.2).
    """
    # Apply OCR corrections first (ligatures, whitespace, char confusions)
    s = preprocess_text_with_ocr_correction(s, context='general')
    
    repls = {
        "☐": "[ ] ", "☑": "[x] ", "□": "[ ] ", "■": "[ ] ", "❒": "[ ] ", "◻": "[ ] ", "◽": "[ ] ",
        "▪": "[ ] ", "•": "• ", "·": "• ", "✓": "[x] ", "✔": "[x] ", "✗": "[ ] ", "✘": "[ ] ",
        "¨": "[ ] ",
    }
    for k, v in repls.items():
        s = s.replace(k, v)
    # Convert standalone "!" to checkbox pattern
    s = re.sub(r"(^|\s)!\s+(?=\w)", r"\1[ ] ", s)
    return s


def collapse_spaced_letters_any(s: str) -> str:
    """
    Collapse spaced-out letters while preserving word boundaries.
    
    Archivev20 Fix 6: Improved spaced letter collapsing
    Pattern: "H o w  d i d  y o u" → "How did you"
    Observation: 1 space between letters within words, 2+ spaces between words
    """
    def collapse_match(match):
        text = match.group(0)
        # Find all letters with their positions
        letters_with_pos = [(m.start(), m.group()) for m in re.finditer(r'[A-Za-z]', text)]
        
        if not letters_with_pos:
            return text
        
        # Build result
        result = [letters_with_pos[0][1]]  # Start with first letter
        
        for i in range(1, len(letters_with_pos)):
            prev_pos = letters_with_pos[i-1][0]
            curr_pos, curr_letter = letters_with_pos[i]
            spaces = curr_pos - prev_pos - 1
            
            # If 2+ spaces, add a word boundary
            if spaces >= 2:
                result.append(' ')
            
            result.append(curr_letter)
        
        return ''.join(result)
    
    s = re.sub(r"(?<!\w)(?:[A-Za-z]\s+){3,}[A-Za-z](?!\w)", collapse_match, s)
    return re.sub(r"\s{2,}", " ", s).strip()


def collapse_spaced_caps(s: str) -> str:
    """Collapse spaced capital letters."""
    s2 = re.sub(r"(?:(?<=\b)|^)(?:[A-Z]\s+){2,}(?=[A-Z]\b)", lambda m: m.group(0).replace(" ", ""), s)
    s2 = collapse_spaced_letters_any(s2)
    return s2.strip()


def read_text_file(p: Path) -> str:
    """
    Read a text file with automatic encoding detection.
    
    Tries UTF-8 first, falls back to latin-1 if that fails.
    Uses 'replace' error handling to avoid encoding errors.
    """
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return p.read_text(encoding="latin-1", errors="replace")


def is_heading(line: str, context: dict = None) -> bool:
    """
    Improvement 10: Enhanced heading detection with context awareness.
    
    Section headings are typically:
    - All uppercase or title case or starts with capital letter
    - <= 120 characters
    - Multi-word descriptive phrases (not single field labels)
    - End with a colon or no punctuation
    - Do NOT contain checkboxes
    - Do NOT match known field label patterns
    - Do NOT contain question marks
    
    Args:
        line: Line to check
        context: Optional dict with:
            - 'has_checkbox': Whether line contains checkbox
            - 'next_line': Next line for underline detection
            - 'original_line': Original line for spacing analysis
            - 'line_position': Position in document
    """
    if context is None:
        context = {}
    
    t = collapse_spaced_caps(line.strip())
    if not t:
        return False
    
    # Archivev10 Fix 1: Don't treat lines with checkboxes as headings
    # Improvement 10: Use context if available
    has_checkbox = context.get('has_checkbox', False) or re.search(CHECKBOX_ANY, t)
    if has_checkbox:
        return False
    
    # PRODUCTION PARITY FIX: Lines with multiple colons are multi-field lines, not headings
    # e.g., "Signature:	Printed Name:	Date:" should be treated as multiple fields
    # Check this BEFORE strong_headers to avoid false matches on "Signature:" alone
    if t.count(':') >= 2:
        return False
    
    # Improvement 10: Strong header indicators (common section names)
    strong_headers = [
        'patient information',
        'medical history',
        'dental history',
        'insurance information',
        'consent',
        'signature',
        'emergency contact',
        'responsible party',
        'health history',
        'treatment information',
        'financial information',
        'dental benefit plan',
        'responsible party information',
    ]
    
    line_lower = t.lower().rstrip(':')
    for header in strong_headers:
        if header == line_lower or line_lower.startswith(header):
            return True
    
    # Improvement 10: All caps and short text (strong header indicator)
    if t.isupper() and 3 <= len(t.split()) <= 5:
        return True
    
    # Improvement 10: Underlined headers (next line is underscores/dashes)
    if context.get('next_line'):
        next_line = context['next_line'].strip()
        if re.match(r'^[_\-=]{5,}$', next_line):
            return True
    
    # Improvement 10: Centered text heuristic (lots of leading spaces)
    if context.get('original_line'):
        original = context['original_line']
        leading_spaces = len(original) - len(original.lstrip())
        if leading_spaces > 10 and len(t.split()) <= 4:
            return True
    
    # Archivev12 Fix: Don't treat known field labels as headings
    # Archivev13 Fix: Use search instead of match, and allow # suffix
    # Check against common form field patterns
    for field_key, pattern in KNOWN_FIELD_LABELS.items():
        if re.search(pattern, t, re.I):
            return False
    
    # Archivev10 Fix 1: Don't treat multi-column grid headers as headings
    # (e.g., "Appearance    Function    Habits    Previous Comfort Options")
    # These have 3+ words/phrases separated by significant spacing
    parts = re.split(r'\s{3,}', t)
    if len(parts) >= 3 and all(len(p.split()) <= 4 for p in parts):
        # Looks like a multi-column grid header, not a section heading
        return False
    
    # Archivev19 Fix 1: Don't treat short single-word field labels as headings
    # e.g., "Comments:", "Notes:", "Explanation:" should be field labels, not headings
    # Section headings are typically multi-word or clearly descriptive
    if t.endswith(":"):
        # Remove the colon and check the remaining text
        label = t[:-1].strip()
        # Single word or two words that look like field labels -> not a heading
        words = label.split()
        if len(words) <= 2:
            # Single word that's not all caps -> likely a field label
            if len(words) == 1 and not label.isupper():
                common_field_labels = ['comments', 'notes', 'explanation', 'details', 'remarks']
                if label.lower() in common_field_labels:
                    return False
            # Two-word phrases that look like field labels (e.g., "Full name:")
            if len(words) == 2:
                common_two_word_labels = ['full name', 'first name', 'last name', 'middle name', 
                                         'phone number', 'email address', 'zip code', 'birth date',
                                         'social security', 'work phone', 'home phone', 'cell phone']
                if label.lower() in common_two_word_labels:
                    return False
    
    # Archivev19 Fix 4: Lines with question marks are questions/fields, never headings
    # e.g., "Have you ever had surgery? If so, what type:" should be a field
    if "?" in t:
        return False
    
    # Enhancement: Lines with underscores/dashes/parentheses are fillable fields, not headings
    # e.g., "Unit ___", "Apt ____", "Phone (   )"
    if re.search(r'_{3,}|[\-]{3,}|\(\s*\)', t):
        return False
    
    # Fix: Accept mixed-case multi-word phrases as section headers
    # Section headers should have multiple words and start with a capital letter
    words = t.rstrip(':').split()
    if len(t) <= 120 and len(words) >= 2:
        # Multi-word phrase starting with capital letter
        if t[0].isupper():
            # Don't end with period (that's a sentence)
            if not t.endswith("."):
                return True
    
    # Original logic: all caps or perfect title case
    if len(t) <= 120 and (t.isupper() or (t.istitle() and not t.endswith("."))):
        if not t.endswith("?"):
            return True
    
    # Don't accept single-word labels ending with colon as headings
    return False


def is_category_header(line: str, next_line: str = "") -> bool:
    """
    Archivev10 Fix 2 + Archivev11 Fix 4: Enhanced category header detection in medical/dental grids.
    Category headers are short lines without checkboxes that precede lines with checkboxes.
    
    Examples: "Cancer", "Cardiovascular", "Endocrinology", "Pain/Discomfort", "Appearance"
    
    Archivev11 Fix 4: Added detection for common label patterns like "Frequency", "Pattern", "Conditions"
    """
    cleaned = collapse_spaced_caps(line.strip())
    
    # Must be reasonably short (category headers or multi-column grid headers)
    if not cleaned or len(cleaned) > 80:
        return False
    
    # Must NOT have checkboxes
    if re.search(CHECKBOX_ANY, cleaned):
        return False
    
    # Must NOT be a question
    if cleaned.endswith("?"):
        return False
    
    # Must NOT end with a colon (that's a field label, not a category header)
    # Examples: "Last Name:", "Work Phone:", "Zip:"
    if cleaned.endswith(":"):
        return False
    
    # Must NOT end with a colon followed by content (that's also a field label)
    if re.search(r':\s*\S', cleaned):
        return False
    
    # Must NOT be a common form field pattern (even without colon)
    # Examples: "Ext#", "Apt#", "SSN", "DOB", "Zip", "State"
    form_field_patterns = [
        r'\b(ext|extension|apt|apartment|ssn|dob|zip|zipcode|state)\s*#?\b',
        r'\b(phone|email|fax|mobile|cell|home|work)\b',
        r'\b(first|last|middle|full)\s+name\b',
    ]
    for pattern in form_field_patterns:
        if re.search(pattern, cleaned, re.I):
            return False
    
    # Archivev11 Fix 4: Check for common label patterns
    # These are often found in forms and should be treated as headers/labels, not fields
    label_keywords = ['frequency', 'pattern', 'conditions', 'health', 'comments', 
                      'how much', 'how long', 'additional comments']
    cleaned_lower = cleaned.lower()
    is_label_pattern = any(kw in cleaned_lower for kw in label_keywords)
    
    # Known category header patterns in medical/dental forms
    category_keywords = [
        'cancer', 'cardiovascular', 'endocrinology', 'musculoskeletal', 
        'respiratory', 'gastrointestinal', 'neurological', 'hematologic',
        'appearance', 'function', 'habits', 'social', 'periodontal',
        'pain', 'discomfort', 'comfort', 'allergies', 'women', 'type',
        'viral infections', 'medical allergies', 'sleep pattern'
    ]
    
    is_known_category = any(kw in cleaned_lower for kw in category_keywords)
    
    # Archivev11 Fix 4: Label patterns with next line having checkboxes are headers
    if is_label_pattern and next_line and re.search(CHECKBOX_ANY, next_line):
        return True
    
    # Next line should have checkboxes (indicates this is a header for checkbox items)
    if next_line and re.search(CHECKBOX_ANY, next_line):
        # Check word count - category headers are usually 1-6 words (or multiple short phrases)
        # E.g., "Appearance Function Habits Previous Comfort Options" = 5 words but valid
        word_count = len(cleaned.split())
        if word_count <= 6 or is_known_category:
            return True
    
    # Also consider it a category header if it's a known category even without next line check
    # (some grids have category headers that span columns)
    if is_known_category and len(cleaned.split()) <= 3:
        return True
    
    return False


def normalize_section_name(raw: str) -> str:
    """
    Normalize a section heading to a standard section name.
    
    Maps various heading text to standard section names like:
    - Patient Information
    - Insurance
    - Medical History
    - etc.
    """
    t = collapse_spaced_caps(raw).strip().lower()
    if "signature" in t:
        return "Signature"
    table = {
        "Patient Information": ["patient information", "patient info", "patient details", "demographic", "registration"],
        "Insurance": ["insurance", "subscriber", "policy", "carrier", "dental benefit plan"],
        "Medical History": ["medical history", "health history", "medical", "medical conditions", "health conditions"],
        "Medications": ["medication", "medications", "current medicines", "rx"],
        "Allergies": ["allergy", "allergies"],
        "Dental History": ["dental history", "dental information", "dental"],
        "HIPAA": ["hipaa"],
        "Financial Policy": ["financial", "payment policy", "billing"],
        "Emergency Contact": ["emergency contact"],
        "Appointment Policy": ["appointment policy", "cancellation", "missed appointment", "late policy"],
        "Consent": ["consent", "authorization", "informed consent", "release"],
    }
    best = ("General", 0)
    for norm, kws in table.items():
        score = sum(1 for k in kws if k in t)
        if score > best[1]:
            best = (norm, score)
    return best[0] if best[1] else "General"


def detect_repeated_lines(lines: List[str], min_count: int = 3, max_len: int = 80) -> Set[str]:
    """
    Detect lines that repeat multiple times (likely headers/footers).
    
    Returns a set of line texts that appear at least min_count times
    and are <= max_len characters.
    """
    counter = Counter([collapse_spaced_caps(l.strip()) for l in lines if l.strip() and len(l.strip()) <= max_len])
    return {s for s, c in counter.items() if c >= min_count}


def is_address_block(block: List[str]) -> bool:
    """
    Check if a block is primarily business/practice address information (not form content).
    
    Returns True only if the block looks like header/footer practice info,
    not if it contains actual form fields.
    """
    # Count different types of content
    address_hits = 0
    form_field_hits = 0
    business_hits = 0
    
    for ln in block:
        ln_lower = ln.lower()
        
        # Check for actual street addresses (with numbers + street type)
        if re.search(r'\b\d+\s+[NS]?\s*\w+\s+(ave|avenue|rd|road|st|street|blvd|boulevard)\b', ln, re.I):
            address_hits += 1
        
        # Check for business/practice names
        if re.search(r'\b(dental|dentistry)\s+(care|center|design|solutions|office)\b', ln, re.I):
            business_hits += 1
        
        # Check for form field labels (labels with colons that indicate form fields)
        if re.search(r'\b(last\s+name|first\s+name|patient\s+name|birth\s+date|dob|address|city|state|zip\s*code?|phone|email|gender|marital|emergency|ssn|insurance)\s*:', ln, re.I):
            form_field_hits += 1
    
    # Only consider it an address block if:
    # 1. It has business/address information AND
    # 2. It has NO form field labels (or very few relative to address content)
    has_business_content = (address_hits >= 2 or business_hits >= 1)
    has_form_content = form_field_hits >= 3
    
    # If it has significant form content, it's not just an address block
    if has_form_content:
        return False
    
    return len(block) >= 3 and has_business_content


def scrub_headers_footers(text: str) -> List[str]:
    """
    Remove headers, footers, and practice information from extracted text.
    
    This function:
    1. Splits text into blocks
    2. Filters out address/business blocks
    3. Removes repeated lines (headers/footers)
    4. Filters out page numbers, practice info, and junk text
    
    Returns a list of cleaned lines.
    """
    raw_lines = text.splitlines()
    blocks: List[List[str]] = []
    cur: List[str] = []
    for ln in raw_lines:
        if ln.strip():
            cur.append(ln)
        else:
            if cur:
                blocks.append(cur); cur=[]
    if cur:
        blocks.append(cur)

    kept_blocks: List[List[str]] = []
    for b in blocks:
        b_trim = [collapse_spaced_caps(x) for x in b]
        if is_address_block(b_trim):
            continue
        kept_blocks.append(b)

    lines = []
    for b in kept_blocks:
        lines.extend(b); lines.append("")

    # Enhanced junk text filtering patterns (Fix 3)
    MULTI_LOCATION_RE = re.compile(
        r'.*\b(Ave|Avenue|St|Street|Rd|Road|Blvd|Boulevard)\.?\b.*\b(Ave|Avenue|St|Street|Rd|Road|Blvd|Boulevard)\.?\b',
        re.I
    )
    CITY_STATE_ZIP_RE = re.compile(r',\s*[A-Z]{2}\s+\d{5}')
    OFFICE_NAMES_RE = re.compile(
        r'\b(dental|care|center|clinic|office|practice)\b.*\b(dental|care|center|clinic|office|practice)\b',
        re.I
    )

    repeats = detect_repeated_lines(lines)
    keep = []
    first_block = True
    block_hits = 0
    form_field_hits = 0  # Count form field indicators
    for ln in lines:
        s = collapse_spaced_caps(ln.strip())
        if s:
            if first_block:
                # Check for actual business addresses (not form field labels)
                # Business addresses have: street name + Ave/Rd/St + city/state pattern
                is_business_address = bool(re.search(r'\b\d+\s+[NS]?\s*\w+\s+(Ave|Avenue|Rd|Road|St|Street|Blvd|Boulevard)\b', s, re.I))
                # Also check for practice names
                is_practice_name = bool(re.search(r'\b(dental|dentistry)\s+(care|center|design|solutions)\b', s, re.I))
                
                if is_business_address or is_practice_name:
                    block_hits += 1
                
                # Count form field indicators (fields with colons that are form labels)
                if re.search(r'\b(name|phone|email|address|city|state|zip|birth|date|ssn|gender|marital)\s*:', s, re.I):
                    form_field_hits += 1
        else:
            if first_block:
                # Only drop first block if it has business addresses AND no form fields
                # This prevents dropping the patient registration section
                if block_hits >= 2 and form_field_hits == 0:
                    # drop first block entirely - it's just header/practice info
                    keep = []
                    first_block = False
                    block_hits = 0
                    form_field_hits = 0
                    continue
                first_block = False
        if not s:
            keep.append(ln); continue
        if s in repeats or PAGE_NUM_RE.match(s): continue
        
        # NEW FILTERS (Fix 3):
        # Filter out lines with multiple street addresses
        if MULTI_LOCATION_RE.search(s):
            continue
        
        # Filter out lines with multiple city-state-zip patterns
        if len(CITY_STATE_ZIP_RE.findall(s)) >= 2:
            continue
        
        # Filter out lines that look like multiple office names
        if OFFICE_NAMES_RE.search(s) and len(s) > 80:
            continue
        
        # Filter out lines with multiple zip codes
        if len(re.findall(r'\b\d{5}\b', s)) >= 2:
            continue
        
        # Archivev8 Fix 2: Enhanced Header/Business Information Filtering
        # Get the line index for top-of-document check
        idx = lines.index(ln) if ln in lines else 999
        
        # Filter lines with dental practice email addresses + business keywords
        if DENTAL_PRACTICE_EMAIL_RE.search(s):
            # Check if line also has practice/business keywords
            if re.search(r'\b(?:dental|dentistry|family|cosmetic|implant|orthodontics)\b', s, re.I):
                continue
        
        # Filter long lines combining business name with address
        if BUSINESS_WITH_ADDRESS_RE.search(s):
            # Additional check: line is quite long (likely a header)
            if len(s) > 50:
                continue
        
        # Filter lines at top of document (first 20 lines) that look like practice headers
        if idx < 20:
            # Check for practice name + address pattern
            has_practice_keyword = bool(re.search(r'\b(?:dental|dentistry|orthodontics|family|cosmetic|implant)\b', s, re.I))
            has_address_keyword = bool(re.search(r'\b(?:suite|ste\.?|ave|avenue|rd|road|st|street|blvd)\b', s, re.I))
            has_contact = bool(re.search(r'(?:@|phone|tel|fax|\d{3}[-.\s]?\d{3}[-.\s]?\d{4})', s, re.I))
            
            # If it has 2+ of these indicators and is long, likely a header
            score = sum([has_practice_keyword, has_address_keyword, has_contact])
            if score >= 2 and len(s) > 40:
                continue
        
        # Existing filters
        if re.search(r"\bcontinued on back side\b", s, re.I): continue
        if re.search(r"\brev\s*\d{1,2}\s*/\s*\d{2}\b", s, re.I): continue
        if s in {"<<<", ">>>"} or re.search(r"\bOC\d+\b", s): continue
        keep.append(ln)
    return keep


def coalesce_soft_wraps(lines: List[str]) -> List[str]:
    """
    Intelligently join lines that were soft-wrapped in the PDF.
    
    Lines are joined if:
    - Previous line ends with hyphen or slash
    - Next line starts with lowercase or small connector word
    - Previous line ends with Yes/No checkboxes and next starts lowercase
    
    Returns a list of lines with soft wraps coalesced.
    """
    out: List[str] = []
    i = 0
    while i < len(lines):
        a = lines[i]
        if not a.strip():
            out.append(a); i += 1; continue
        merged = a.rstrip()
        while i + 1 < len(lines):
            b = lines[i+1]
            b_str = b.strip()
            if not b_str: break
            if is_heading(b_str): break
            if BULLET_RE.match(b_str): break
            a_end = merged[-1] if merged else ""
            starts_lower = bool(re.match(r"^[a-z(]", b_str))
            small_word  = bool(re.match(r"^(and|or|if|but|then|with|of|for|to)\b", b_str, re.I))
            
            # Enhanced line coalescing (Fix 5):
            # More aggressive continuation for incomplete questions
            ends_with_question = a_end == "?"
            starts_with_paren = b_str.startswith("(")
            
            # Archivev19 Fix 2: Handle multi-line questions where line 1 ends with "/ [ ] Yes [ ] No"
            # and line 2 starts with lowercase continuation (e.g., bisphosphonates question)
            # Pattern: "...Actonel/ [ ] Yes [ ] No" followed by "other medications..."
            ends_with_yes_no = bool(re.search(r'/\s*\[\s*\]\s*(?:Yes|No)\s*(?:\[\s*\]\s*(?:Yes|No)\s*)?$', merged, re.I))
            
            # Join if: 
            # 1. hyphen/slash at end, OR
            # 2. (not sentence-ending punctuation AND (starts lowercase OR small word OR starts with paren)), OR
            # 3. Archivev19: ends with Yes/No checkboxes and next line starts with lowercase (continuation)
            if (a_end in "-/" or 
                (not ends_with_question and a_end not in ".:;?!" and (starts_lower or small_word or starts_with_paren)) or
                (ends_with_yes_no and starts_lower)):
                merged = (merged.rstrip("- ") + " " + b_str).strip()
                i += 1; continue
            break
        out.append(merged)
        i += 1
    return out


def is_numbered_list_item(line: str) -> bool:
    """
    NEW Improvement 1: Detect if line is a numbered list item that should be part of Terms/consent.
    
    Consent forms contain numbered risk/benefit lists like "(i)", "(ii)", "(iii)", etc.
    These should not be parsed as individual input fields but rather be part of parent consent text.
    
    Patterns detected:
    - (i), (ii), (iii), ..., (xxx) - Roman numerals in parentheses
    - (1), (2), (3) - Arabic numerals in parentheses
    - i), ii), iii) - Roman numerals with closing parenthesis only
    - 1., 2., 3. - Numbered items (common in risk lists)
    - Usually followed by lowercase text (continuation of list)
    
    Args:
        line: Line to check
        
    Returns:
        True if line appears to be a numbered list item from consent text
    """
    if not line:
        return False
    
    # Match patterns at start of line
    list_patterns = [
        r'^\s*\([ivxlcdm]+\)\s+[a-z]',  # (i) lowercase continuation
        r'^\s*\(\d+\)\s+[a-z]',          # (1) lowercase continuation
        r'^\s*[ivxlcdm]+\)\s+[a-z]',    # i) lowercase continuation
        # Also match even without lowercase after (for edge cases)
        r'^\s*\([ivxlcdm]{1,4}\)\s*[a-z]', # (i), (ii), (iii), (iv) etc
        r'^\s*\(\d{1,2}\)\s*[a-z]',        # (1), (2), ..., (99)
        # Numbered list items like "1. Risk item", "2. Another risk"
        r'^\s*\d{1,2}\.\s+[A-Z]',          # 1. Capital letter (risk/benefit lists)
    ]
    
    line_lower = line.lower().strip()
    for pattern in list_patterns:
        if re.match(pattern, line_lower):
            return True
    
    return False


def is_instructional_paragraph(line: str) -> bool:
    """
    Improvement #7: Detect if line is an instructional paragraph (not a form field).
    
    Instructional/consent text has these characteristics:
    - Long sentences (>50 words or >250 characters)
    - Multiple sentences (contains 2+ periods)
    - Starts with consent phrases ("I understand", "I authorize", "I certify")
    - Contains legal/medical terminology
    - No fill-in blanks (underscores, parentheses)
    
    Args:
        line: Line to check
        
    Returns:
        True if line appears to be instructional text, not a field
    """
    if not line or len(line) < 30:
        return False
    
    line_stripped = line.strip()
    
    # Count words
    word_count = len(line_stripped.split())
    
    # CRITICAL FIX: Before classifying long lines as instructional, check for form field indicators
    # Long lines with many underscores or multiple colons are likely concatenated form fields
    underscore_count = line_stripped.count('_')
    colon_count = line_stripped.count(':')
    
    # If line has many underscores (3+ sets of 3+ underscores) or many colons (5+), it's likely form fields
    underscore_sequences = len(re.findall(r'_{3,}', line_stripped))
    if underscore_sequences >= 3 or colon_count >= 5:
        return False  # This is likely a form field line, not instructional text
    
    # PRODUCTION PARITY FIX: Check for embedded parenthetical field labels
    # Lines like "PATIENT CONSENT: I, _____(print name) have been..." contain fillable fields
    # Pattern: underscores followed by parenthetical label
    if re.search(r'_{3,}\s*\([^)]{3,40}\)', line_stripped):
        return False  # This line contains a fillable field, not just instructions
    
    # Long text is likely instructional (>50 words or >250 chars)
    # BUT only if it doesn't have form field indicators (checked above)
    if word_count > 50 or len(line_stripped) > 250:
        return True
    
    # Multiple sentences (2+ periods, excluding abbreviations)
    period_count = line_stripped.count('.')
    # Exclude common abbreviations
    abbrev_count = len(re.findall(r'\b(?:Dr|Mr|Mrs|Ms|Inc|Ltd|Jr|Sr|etc)\.\b', line_stripped, re.I))
    if (period_count - abbrev_count) >= 2:
        return True
    
    # Strong consent/instructional phrase indicators
    instructional_starts = [
        r'^i\s+(?:hereby\s+)?(?:understand|certify|acknowledge|consent|agree|authorize|give)',
        r'^(?:the\s+)?patient\s+(?:understands?|acknowledges?|consents?|agrees?)',
        r'^by\s+signing',
        r'^i\s+have\s+(?:read|been\s+(?:informed|given))',
        r'^(?:this|it)\s+is\s+(?:understood|acknowledged)',
        r'^we\s+(?:understand|acknowledge)',
        # Add imperative instruction patterns
        r'^do\s+not\s+(?:consume|take|eat|drink|smoke)',
        r'^please\s+(?:arrive|bring|remove|wear|leave)',
        r'^ensure\s+(?:you|that)',
        r'^your\s+escort\s+must',
        r'^we\s+reserve\s+the\s+right',
        r'^if\s+you\s+do\s+not',
        # General imperative patterns (verb at start followed by object)
        r'^(?:take|wear|remove|bring|arrive)\s+(?:any|your|all|the|short)',
        # Consent form references (Archivev23)
        r'^this\s+(?:consent\s+)?form\s+(?:should|must|is|will)',
        r'^this\s+(?:document|consent)\s+(?:should|must|is|will)',
    ]
    
    line_lower = line_stripped.lower()
    for pattern in instructional_starts:
        if re.match(pattern, line_lower):
            # If it starts with these AND is reasonably long, it's instructional
            # Lower threshold for imperative instructions (6+ words)
            # Even lower (10+ words) for "this form/document" patterns
            if word_count > 10 or (word_count > 6 and 'form' not in pattern):
                return True
    
    # Legal/risk terminology in longer sentences
    if word_count > 20:
        risk_terms = [
            'may include', 'may result in', 'possible risks', 'complications',
            'not limited to', 'include but', 'such as', 'alternative treatment',
            'necessitating', 'have been explained', 'I was able to'
        ]
        if any(term in line_lower for term in risk_terms):
            return True
    
    # Explanatory phrases
    if word_count > 25:
        explanatory = [
            'this means that', 'it is important to', 'you should', 'please note',
            'be aware that', 'keep in mind', 'it is your responsibility'
        ]
        if any(phrase in line_lower for phrase in explanatory):
            return True
    
    return False


def is_form_metadata(line: str) -> bool:
    """
    NEW Improvement 6: Detect if line is form metadata that should be filtered out.
    
    Form identifiers, revision codes, and copyright text should not become fields.
    
    Patterns detected:
    - Revision codes: "REV A", "F16015_REV_E", "v1.0", "Version 2.1"
    - Copyright: "All rights reserved", "© 2024", "Copyright"
    - Contact info: Phone numbers, websites at line boundaries
    - Form codes: Alphanumeric codes like "F16015"
    - Company boilerplate: Company names with "Inc", "LLC", etc.
    
    Args:
        line: Line to check
        
    Returns:
        True if line appears to be form metadata
    """
    if not line or len(line) < 3:
        return False
    
    # Patterns that indicate metadata
    metadata_patterns = [
        r'\brev\s*[a-z0-9]\b',                    # REV A, REV E, REV 1
        r'[a-z]\d{4,6}_rev_[a-z0-9]',            # F16015_REV_E
        r'\bv\d+\.\d+\b',                        # v1.0, v2.1
        r'\bversion\s+\d+',                      # Version 1, Version 2.1
        r'all\s+rights\s+reserved',              # Copyright text
        r'©|copyright\s+\d{4}',                  # Copyright symbols/year
        r'^\s*\(\d{3}\)\s*\d{3}-\d{4}\s*$',     # Standalone phone numbers
        r'www\.\w+\.com',                        # Websites
        r'^\s*[a-z]\d{4,6}\s*$',                # Form codes alone on line
        r'\b(inc|llc|ltd|corp|corporation)\b.*\(\d{3}\)',  # Company with phone
        r'align\s+technology.*inc',              # Specific company names
        r'^\s*[a-z0-9]{6,}_[a-z]+_[a-z]\s*$',   # Codes like "F16015_REV_E"
    ]
    
    line_lower = line.lower().strip()
    
    for pattern in metadata_patterns:
        if re.search(pattern, line_lower):
            return True
    
    # Additional heuristic: very short alphanumeric codes (likely form IDs)
    # But not if it contains common words
    if len(line_lower) <= 12 and re.match(r'^[a-z0-9_\-]+$', line_lower):
        # Check if it's not a common abbreviation or word
        common_words = ['yes', 'no', 'other', 'name', 'date', 'phone', 'email', 'city', 'state', 'zip', 'witness', 'signature']
        if line_lower not in common_words:
            # Could be a form code
            return True
    
    return False


def is_practice_location_text(line: str, context: list = None) -> bool:
    """
    NEW Improvement 7: Detect if line is practice/office location information.
    
    Office addresses and location names embedded in forms should not become fields.
    
    Indicators:
    - Contains "Dental" + address components
    - Multiple consecutive lines with addresses  
    - Pattern: "Name + Street + City, State ZIP"
    - Common dental practice keywords
    
    Args:
        line: Line to check
        context: Previous lines for multi-address detection
        
    Returns:
        True if line appears to be practice location information
    """
    if not line or len(line) < 10:
        return False
    
    if context is None:
        context = []
    
    # Common practice name keywords
    dental_practice_keywords = [
        'dental care',
        'dental center',
        'dental solutions',
        'dental office',
        'dental group',
        'dental associates',
        'dentistry',
        'orthodontics',
        'oral surgery',
    ]
    
    # Address patterns
    address_patterns = [
        r'\d+\s+[NSEW]\.?\s+\w+\s+(ave|avenue|st|street|rd|road|blvd|boulevard|dr|drive|ln|lane|way|ct|court)',  # Street address
        r',\s*[A-Z]{2}\s+\d{5}',                       # City, ST ZIP
        r'\d{5}(-\d{4})?$',                            # ZIP code at end
    ]
    
    line_lower = line.lower()
    
    # Check if line has dental keyword + address
    has_dental_keyword = any(kw in line_lower for kw in dental_practice_keywords)
    has_address = any(re.search(p, line, re.I) for p in address_patterns)
    
    if has_dental_keyword and has_address:
        return True
    
    # Check if surrounded by similar address lines (multi-location forms)
    # This helps catch address-only lines in between practice names
    if context and len(context) >= 1:
        context_has_addresses = sum(
            any(re.search(p, ctx, re.I) for p in address_patterns)
            for ctx in context[-2:] if ctx
        )
        # If previous lines had addresses and this line has an address, likely continuation
        if context_has_addresses >= 1 and has_address:
            return True
    
    # Check if line is mostly an address (street number + street name pattern)
    if re.search(r'^\s*\d+\s+[NSEW]\.?\s+\w+', line):
        return True
    
    return False


def separate_field_label_from_blanks(line: str) -> str:
    """
    Improvement #1: Separate field labels from underscore/blank patterns.
    
    Transforms patterns like "Label____" into "Label: ___" for better parsing.
    This helps the parser distinguish the label from the fill-in area.
    
    Patterns handled:
    - "Label____" → "Label: ___"
    - "Label:____" → "Label: ___" (normalize)
    - "Label (___)" → "Label: ___"
    - "Label [___]" → "Label: ___"
    
    Args:
        line: Input line with potential label+blank patterns
        
    Returns:
        Normalized line with separated labels and blanks
    """
    if not line or '___' not in line:
        return line
    
    # Pattern 1: Label immediately followed by underscores (no colon)
    # "Name____" → "Name: ___"
    line = re.sub(r'([A-Za-z][A-Za-z\s]{1,30})_{3,}', r'\1: ___', line)
    
    # Pattern 2: Label with colon immediately followed by underscores (no space)
    # "Name:____" → "Name: ___"
    line = re.sub(r'([A-Za-z][A-Za-z\s]{1,30}):_{3,}', r'\1: ___', line)
    
    # Pattern 3: Label with parentheses/brackets around blanks
    # "Name (___)" → "Name: ___"
    line = re.sub(r'([A-Za-z][A-Za-z\s]{1,30})\s*[\(\[]_{3,}[\)\]]', r'\1: ___', line)
    
    # Pattern 4: Multiple consecutive blank areas - normalize spacing
    # "___  ___  ___" → "___"
    line = re.sub(r'_{3,}\s*_{3,}\s*_{3,}', '___', line)
    
    # Pattern 5: Clean up any double colons created
    line = re.sub(r'::\s*', ': ', line)
    
    return line


def normalize_compound_field_line(line: str) -> List[str]:
    """
    Improvement #5: Split compound field lines into separate field labels.
    
    Handles patterns like:
    - "First Name: ___ MI: ___ Last Name: ___"
    - "Phone: Mobile ___ Home ___ Work ___"
    - "City: ___ State: ___ ZIP: ___"
    
    Returns a list of separate field lines if splitting is appropriate,
    otherwise returns a single-item list with the original line.
    
    Args:
        line: Input line that may contain multiple fields
        
    Returns:
        List of field strings (split or original)
    """
    if not line or line.count(':') < 2:
        return [line]
    
    # First, separate labels from blanks if needed
    line = separate_field_label_from_blanks(line)
    
    # Pattern: Label: blank_area Label: blank_area
    # Match labels that are followed by colon and then optional blanks/content
    pattern = r'([A-Z][A-Za-z\s]{1,30}):\s*([_\(\)\[\]]*)'
    
    matches = list(re.finditer(pattern, line))
    
    # Need at least 2 matches to split
    if len(matches) < 2:
        return [line]
    
    # Don't split if line looks like a question (has '?')
    if '?' in line:
        return [line]
    
    # Don't split if total length is very short (likely not compound fields)
    if len(line) < 30:
        return [line]
    
    fields = []
    for i, match in enumerate(matches):
        label = match.group(1).strip()
        blank_area = match.group(2).strip()
        
        # Skip very short labels
        if len(label) < 2:
            continue
        
        # Extract content between this match and next (or to end)
        start_pos = match.end()
        if i < len(matches) - 1:
            end_pos = matches[i + 1].start()
        else:
            end_pos = len(line)
        
        # Get any text/content after the blank
        between_content = line[start_pos:end_pos].strip()
        
        # Build field string
        if blank_area:
            field_str = f"{label}: {blank_area}"
        elif between_content and not re.match(pattern, between_content):
            # Content after label but before next label
            field_str = f"{label}: {between_content}"
        else:
            field_str = f"{label}: ___"
        
        fields.append(field_str.strip())
    
    # Only return split fields if we got at least 2 meaningful ones
    return fields if len(fields) >= 2 else [line]
