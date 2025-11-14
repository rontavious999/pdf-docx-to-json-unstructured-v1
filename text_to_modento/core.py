#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
text_to_modento.py — v2.21

TXT (Unstructured) -> Modento-compliant JSON

What's new vs v2.20 (Priority Improvements):
  • Priority 1.1: OCR Auto-Detection - Automatically detects and processes scanned PDFs
  • Priority 2.1: Multi-Field Label Splitting - Enhanced detection of multi-subfield patterns
  • Priority 2.2: Grid Column Headers - Option A implementation (prefix approach)
  • Priority 2.3: Inline Checkbox Detection - Enhanced mid-sentence checkbox recognition
  • Fixed module exports for proper test compatibility
  • All implementations use generic patterns - NO form-specific hardcoding
  
Previous versions:
  • v2.20/Archivev19 Fix 4: Never treat lines with question marks as section headings
  • v2.19/Archivev19 Fix 3: Inline checkbox field title extraction preserves labels
  • v2.18/Archivev19 Fix 1-2: Single-word field labels, multi-line questions

Patch 2: Incremental Modularization (Evaluation Feedback)
---------------------------------------------------------
This file is being incrementally refactored to improve maintainability.

Current Status:
  ✓ Text preprocessing → modules/text_preprocessing.py
  ✓ Grid parsing → modules/grid_parser.py
  ✓ Template matching → modules/template_catalog.py
  ✓ Basic utilities → modules/question_parser.py
  ✓ Debug logging → modules/debug_logger.py
  
Planned Modularization (Future PRs):
  □ Field detection functions → modules/field_detection.py (~500 lines)
  □ Postprocessing functions → modules/postprocessing.py (~770 lines)
  □ Validation functions → modules/validation.py (~100 lines)
  
Target: Reduce core.py to ~1500 lines of orchestration logic.

Current Structure (4135 lines):
  1. Imports and constants (lines 1-285)
  2. Field splitting and detection functions (lines 286-1960)
  3. Main parsing logic (lines 1961-3030)
  4. Validation and deduplication (lines 3031-3100)
  5. Postprocessing functions (lines 3101-3870)
  6. Template application and I/O (lines 3871-4100)
  7. Main entry point (lines 4101-4135)
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from difflib import SequenceMatcher
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

# Import from modularized components
from .modules.text_preprocessing import (
    normalize_glyphs_line,
    collapse_spaced_letters_any,
    collapse_spaced_caps,
    read_text_file,
    is_heading,
    is_category_header,
    normalize_section_name,
    detect_repeated_lines,
    is_address_block,
    scrub_headers_footers,
    coalesce_soft_wraps,
    is_numbered_list_item,  # NEW Improvement 1
    is_form_metadata,  # NEW Improvement 6
    is_practice_location_text,  # NEW Improvement 7
    is_instructional_paragraph,  # Improvement #7
    separate_field_label_from_blanks,  # Improvement #1
    normalize_compound_field_line  # Improvement #5
)
from .modules.grid_parser import (
    looks_like_grid_header,
    detect_table_layout,
    parse_table_to_questions,
    chunk_by_columns,
    detect_column_boundaries,
    detect_multicolumn_checkbox_grid,
    parse_multicolumn_checkbox_grid,
    extract_text_for_checkbox,
    extract_text_only_items_at_columns,
    detect_medical_conditions_grid
)
from .modules.template_catalog import (
    TemplateCatalog,
    FindResult,
    merge_with_template,
    _dedupe_keys_dicts,
    _norm_text,
    _slug_key_norm,
    _token_set_ratio,
    _is_conditions_control,
    _sanitize_words_set,
    _alias_tokens_ok,
    _sanitize_words
)
# Patch 7 Phase 1: Import utility functions from question_parser
from .modules.question_parser import (
    slugify,
    clean_token,
    normalize_opt_name,
    clean_option_text,
    clean_field_title,
    make_option,
    classify_input_type,
    classify_date_input,
    norm_title,  # Patch 2: Moved to question_parser for better organization
    generate_contextual_date_key,  # Improvement 6: Date field disambiguation
    infer_multi_select_from_context,  # NEW Improvement 8: Smart multi-select detection
    recognize_semantic_field_label,  # Improvement #4
    detect_empty_vs_filled_field,  # Improvement #14
    infer_field_context_from_section,  # Improvement #12
)
# Parity Improvements: Import field detection and consent handling modules
from .modules.field_detection import (
    split_colon_delimited_fields,
    split_multi_subfield_line,
    should_split_line_into_fields,
    normalize_checkbox_symbols,
    infer_field_type_from_label,
)
from .modules.consent_handler import (
    is_consent_paragraph,
    is_consent_section_header,
    group_consecutive_consent_paragraphs,
    is_risk_list_header,
    group_risk_list_items,
    normalize_signature_field,
    parse_tabulated_signature_line,
    is_tabulated_signature_line,
)
# Improvement #10, #11, #15: Import postprocessing enhancements
from .modules.postprocessing import (
    consolidate_duplicate_fields_enhanced,
    infer_section_boundaries,
    calculate_field_confidence,
    add_confidence_scores,
    filter_low_confidence_fields,
)
# Improvement #2: Enhanced OCR correction
from .modules.ocr_correction import (
    enhance_dental_term_corrections,
    correct_phone_number_patterns,
    correct_date_patterns,
    clean_checkbox_ocr_artifacts,
)
# Performance Recommendations: Enhanced detection and consent handling
from .modules.performance_enhancements import (
    detect_inline_checkbox_options,
    enhance_field_type_detection,
    consolidate_procedural_consent_blocks,
    is_procedural_consent_text,
)

# ---------- Paths

DEFAULT_IN_DIR = "output"
DEFAULT_OUT_DIR = "JSONs"

# ---------- Patch 4: Template Catalog Caching
# Cache the loaded catalog per worker process to avoid repeated I/O
_loaded_catalog = None

def get_template_catalog(path: Path) -> Optional[TemplateCatalog]:
    """
    Get or load the template catalog with caching.
    
    Patch 4: Performance optimization for parallel processing
    - Loads dictionary once per worker process, not once per file
    - Reduces redundant disk I/O and JSON parsing
    - Each worker process has its own cached copy (process-safe)
    
    Args:
        path: Path to the dental_form_dictionary.json file
        
    Returns:
        TemplateCatalog instance or None if loading fails
    """
    global _loaded_catalog
    if _loaded_catalog is None:
        try:
            _loaded_catalog = TemplateCatalog.from_path(path)
        except Exception:
            return None
    return _loaded_catalog

# ---------- Regex / tokens

CHECKBOX_ANY = r"(?:\[\s*\]|\[x\]|☐|☑|□|■|❒|◻|✓|✔|✗|✘)"
BULLET_RE = re.compile(r"^\s*(?:[-*•·]|" + CHECKBOX_ANY + r")\s+")
CHECKBOX_MARK_RE = re.compile(r"^\s*(" + CHECKBOX_ANY + r")\s+")

INLINE_CHOICE_RE = re.compile(
    rf"(?:^|\s){CHECKBOX_ANY}\s*([^\[\]•·\-\u2022]+?)(?=(?:\s*{CHECKBOX_ANY}|\s*[•·\-]|$))"
)

YESNO_SET = {"yes", "no", "y", "n"}

PHONE_RE   = re.compile(r"\b(phone|cell|mobile|telephone)\b", re.I)
EMAIL_RE   = re.compile(r"\bemail\b", re.I)
ZIP_RE     = re.compile(r"\b(zip|postal)\b", re.I)
SSN_RE     = re.compile(r"\b(ssn|social security|soc(?:ial)?\s*sec(?:urity)?|ss#)\b", re.I)
STATE_LABEL_RE = re.compile(r"^\s*state\b", re.I)
DATE_LABEL_RE  = re.compile(r"\b(date|dob|birth)\b", re.I)
INITIALS_RE    = re.compile(r"\binitials?\b", re.I)
WITNESS_RE     = re.compile(r"\bwitness\b", re.I)

IF_GUIDANCE_RE = re.compile(r"\b(if\s+(yes|no)[^)]*|if\s+so|if\s+applicable|explain below|please explain|please list)\b", re.I)

# Enhanced "If Yes" detection patterns (Fix 2)
IF_YES_FOLLOWUP_RE = re.compile(
    r'(.+?)\s*\[\s*\]\s*Yes\s*\[\s*\]\s*No\s+If\s+yes[,:]?\s*(?:please\s+)?explain',
    re.I
)
IF_YES_INLINE_RE = re.compile(
    r'(.+?)\s*\[\s*\]\s*Yes\s*\[\s*\]\s*No\s+If\s+yes',
    re.I
)

PAGE_NUM_RE = re.compile(r"^\s*(?:page\s*\d+(?:\s*/\s*\d+)?|\d+\s*/\s*\d+)\s*$", re.I)
ADDRESS_LIKE_RE = re.compile(
    r"\b(?:street|suite|ste\.?|ave|avenue|rd|road|blvd|zip|postal|fax|tel|phone|www\.|https?://|\.com|\.net|\.org|welcome|new\s+patients)\b",
    re.I,
)

# Enhanced header filtering patterns (Archivev8 Fix 2)
DENTAL_PRACTICE_EMAIL_RE = re.compile(
    r'@.*?(?:dental|dentistry|orthodontics|smiles).*?\.(com|net|org)',
    re.I
)
BUSINESS_WITH_ADDRESS_RE = re.compile(
    r'(?:dental|dentistry|orthodontics|family|cosmetic|implant).{20,}?(?:suite|ste\.?|ave|avenue|rd|road|st\.?|street)',
    re.I
)
PRACTICE_NAME_PATTERN = re.compile(
    r'^(?:.*?(?:dental|dentistry|orthodontics|family|cosmetic|implant).*?){1,3}$',
    re.I
)

INSURANCE_PRIMARY_RE   = re.compile(r"\bprimary\b.*\binsurance\b", re.I)
INSURANCE_SECONDARY_RE = re.compile(r"\bsecondary\b.*\binsurance\b", re.I)
INSURANCE_BLOCK_RE     = re.compile(r"(dental\s+benefit\s+plan|insurance\s+information|insurance\s+details)", re.I)

SINGLE_SELECT_TITLES_RE = re.compile(r"\b(marital\s+status|relationship|gender|sex)\b", re.I)

HEAR_ABOUT_RE   = re.compile(r"how\s+did\s+you\s+hear\s+about\s+us", re.I)
REFERRED_BY_RE  = re.compile(r"\b(referred\s+by|who\s+can\s+we\s+thank)\b", re.I)

RESP_PARTY_RE   = re.compile(r"responsible\s+party.*other\s+than\s+patient", re.I)
SINGLE_BOX_RE   = re.compile(r"^\s*\[\s*\]\s*(.+)$")

# broader Y/N capture (no boxes)
YN_SIMPLE_RE = re.compile(r"(?P<prompt>.*?)(?:\bYes\b|\bY\b)\s*(?:/|,|\s+)\s*(?:\bNo\b|\bN\b)", re.I)

# parent/guardian
PARENT_RE = re.compile(r"\b(parent|guardian|mother|father|legal\s+guardian)\b", re.I)

# Archivev12 Fix 2: Special field patterns for common fields without perfect formatting
# Phase 4 Fix 1: Enhanced patterns to detect checkbox-based Sex/Gender and Marital Status fields
SEX_GENDER_PATTERNS = [
    re.compile(r'\b(sex|gender)\s*[:\-]?\s*(?:M\s*or\s*F|M/F|Male/Female)', re.I),
    re.compile(r'\b(sex|gender)\s*\[\s*\]\s*(?:male|female|M|F)', re.I),
    # New: Match "Sex □ Male □ Female" pattern with checkbox characters
    re.compile(r'\bsex\s*' + CHECKBOX_ANY + r'\s*male\s*' + CHECKBOX_ANY + r'\s*female', re.I),
    re.compile(r'\bgender\s*' + CHECKBOX_ANY + r'\s*male\s*' + CHECKBOX_ANY + r'\s*female', re.I),
]

MARITAL_STATUS_PATTERNS = [
    re.compile(r'(?:please\s+)?circle\s+one\s*:?\s*(single|married|divorced|separated|widowed)', re.I),
    re.compile(r'\bmarital\s+status\s*:?\s*(?:\[\s*\]|single|married)', re.I),
    # New: Match "Marital Status □ Married □ Single..." pattern with checkboxes
    re.compile(r'\bmarital\s+status\s*' + CHECKBOX_ANY, re.I),
]

# Phase 4 Fix 3: Pattern for "Preferred method of contact" fields
PREFERRED_CONTACT_PATTERNS = [
    re.compile(r'(?:what\s+is\s+your\s+)?preferred\s+method\s+of\s+contact', re.I),
    re.compile(r'preferred\s+contact\s+method', re.I),
]

PHONE_PATTERNS = [
    (r'work\s+phone', 'work_phone'),
    (r'home\s+phone', 'home_phone'),
    (r'(?:cell|mobile)\s+phone', 'cell_phone'),
]

ALLOWED_TYPES = {"input", "date", "states", "radio", "dropdown", "checkbox", "terms", "signature", "block_signature"}

PRIMARY_SUFFIX = "__primary"
SECONDARY_SUFFIX = "__secondary"

# Medical condition tokens for consolidation detection
_COND_TOKENS = {"diabetes","arthritis","rheumat","hepatitis","asthma","stroke","ulcer",
                "thyroid","cancer","anemia","glaucoma","osteoporosis","seizure","tb","tuberculosis",
                "hiv","aids","blood","pressure","heart","kidney","liver","bleeding","sinus",
                "smoke","chew","alcohol","drug","allergy","pregnan","anxiety","depression","pacemaker",
                "cholesterol","radiation","chemotherapy","convulsion","epilepsy","migraine","valve",
                "neurological","alzheimer"}

# ---------- Debug/Reporting

@dataclass
class MatchEvent:
    title: str
    parsed_key: str
    section: str
    matched_key: Optional[str]
    reason: str
    score: float
    coverage: float

class DebugLogger:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.events: List[MatchEvent] = []
        self.near_misses: List[MatchEvent] = []
        self.gates: List[str] = []

    def log(self, ev: MatchEvent):
        if self.enabled:
            self.events.append(ev)

    def log_near(self, ev: MatchEvent):
        if self.enabled:
            self.near_misses.append(ev)

    def gate(self, msg: str):
        if self.enabled:
            self.gates.append(msg)

    def print_summary(self):
        if not self.enabled:
            return
        print("  [debug] template matches:")
        for ev in self.events:
            print(f"    ✓ '{ev.title}' -> {ev.matched_key} ({ev.reason}, score={ev.score:.2f}, cov={ev.coverage:.2f})")
        if self.near_misses:
            print("  [debug] near-misses (score>=0.75 but rejected):")
            for ev in self.near_misses:
                print(f"    ~ '{ev.title}' -> {ev.matched_key or '—'} ({ev.reason}, score={ev.score:.2f}, cov={ev.coverage:.2f})")
        if self.gates:
            print("  [debug] gates:")
            for g in self.gates:
                print(f"    • {g}")

# ---------- Normalization helpers

SPELL_FIX = {
    "rregular": "Irregular",
    "hyploglycemia": "Hypoglycemia",
    "diabates": "Diabetes",
    "osteoperosis": "Osteoporosis",
    "artritis": "Arthritis",
    "rheurnatism": "Rheumatism",
    "e": "Email",
}

# Text preprocessing functions now imported from modules.text_preprocessing
# These functions handle line cleanup, normalization, and soft-wrap coalescing:
# - normalize_glyphs_line, collapse_spaced_letters_any, collapse_spaced_caps
# - read_text_file, is_heading, is_category_header, normalize_section_name
# - detect_repeated_lines, is_address_block, scrub_headers_footers, coalesce_soft_wraps

# ============================================================================
# SECTION 2: FIELD SPLITTING AND DETECTION FUNCTIONS
# ============================================================================
# These functions split multi-question lines and detect specific field types.
# Future PR: Move to modules/field_detection.py (~500 lines)

# ---------- Fix 1: Split Multi-Question Lines

def split_multi_question_line(line: str) -> List[str]:
    """
    Split lines containing multiple independent questions into separate lines.
    
    Example:
        Input:  "Gender: [ ] Male [ ] Female     Marital Status: [ ] Married [ ] Single"
        Output: ["Gender: [ ] Male [ ] Female", "Marital Status: [ ] Married [ ] Single"]
    
    Detection criteria:
    - Line contains 2+ question labels
    - Significant spacing (4+ spaces) separates the questions
    - Each segment should have checkboxes
    
    Returns:
        List of question strings (original line if no split needed)
    """
    # Strategy: Look for patterns like "...] ... 4+spaces ... Label:"
    # This finds where one question ends (with ]) and another begins (with Label:)
    
    # Pattern: closing bracket, some text/spaces, then 4+ spaces, then a label with colon
    split_pattern = r'\]\s+([^\[]{0,50}?)\s{4,}([A-Z][A-Za-z\s]+?):\s*\['
    matches = list(re.finditer(split_pattern, line))
    
    if not matches:
        # No clear split points found
        return [line]
    
    # Build segments by splitting at the match positions
    segments = []
    last_end = 0
    
    for match in matches:
        # The split point is right before the label (group 2)
        split_pos = match.start() + len(match.group(0)) - len(match.group(2)) - 1 - len(': [')
        
        # Add the segment from last_end to split_pos
        segment = line[last_end:split_pos].strip()
        if segment and re.search(CHECKBOX_ANY, segment):
            segments.append(segment)
        
        last_end = split_pos
    
    # Add the final segment
    final_segment = line[last_end:].strip()
    if final_segment and re.search(CHECKBOX_ANY, final_segment):
        segments.append(final_segment)
    
    # Return segments if we successfully split, otherwise original line
    return segments if len(segments) >= 2 else [line]


# ---------- Archivev12 Fix 1: Enhanced Multi-Field Line Splitting

# Known field labels dictionary for pattern matching
# Production Improvement: Use lookahead to handle underscores/dashes/punctuation after labels
# (?=[^a-zA-Z]|$) means "followed by non-letter or end of string"
KNOWN_FIELD_LABELS = {
    # Name fields
    'full_name': r'\bfull\s+name(?=[^a-zA-Z]|$)',
    'first_name': r'\bfirst\s+name(?=[^a-zA-Z]|$)',
    'last_name': r'\blast\s+name(?=[^a-zA-Z]|$)',
    'preferred_name': r'\bpreferred\s+name(?=[^a-zA-Z]|$)',
    'middle_initial': r'\b(?:middle\s+initial|m\.?i\.?)(?=[^a-zA-Z]|$)',
    'patient_name': r'\b(?:patient(?:\'?s)?\s+name|name\s+of\s+patient)(?=[^a-zA-Z]|$)',
    'parent_name': r'\bparent\s+name(?=[^a-zA-Z]|$)',
    'guardian_name': r'\bguardian\s+name(?=[^a-zA-Z]|$)',
    # Date/Age fields
    'birth_date': r'\b(?:birth\s+date|date\s+of\s+birth|birthdate)(?=[^a-zA-Z]|$)',
    'dob': r'\bdob(?=[^a-zA-Z]|$)',
    'age': r'\bage(?=[^a-zA-Z]|$)',
    'mother_dob': r"\bmother'?s?\s+dob(?=[^a-zA-Z]|$)",
    'father_dob': r"\bfather'?s?\s+dob(?=[^a-zA-Z]|$)",
    # Demographics
    'sex': r'\bsex(?=[^a-zA-Z]|$)',
    'gender': r'\bgender(?=[^a-zA-Z]|$)',
    'marital_status': r'\b(?:marital\s+status|please\s+circle\s+one)(?=[^a-zA-Z]|$)',
    # Contact fields
    'phone_number': r'\bphone\s+number(?=[^a-zA-Z]|$)',
    'work_phone': r'\bwork\s+phone(?=[^a-zA-Z]|$)',
    'home_phone': r'\bhome\s+phone(?=[^a-zA-Z]|$)',
    'cell_phone': r'\b(?:cell|mobile)\s+phone(?=[^a-zA-Z]|$)',
    'parent_phone': r'\bparent\s+phone(?=[^a-zA-Z]|$)',
    'email': r'\be-?mail(?:\s+address)?(?=[^a-zA-Z]|$)',
    'emergency_contact': r'\bemergency\s+contact(?=[^a-zA-Z]|$)',
    'phone': r'\bphone(?=[^a-zA-Z]|$)',
    'ext': r'\bext\s*#?(?=[^a-zA-Z]|$)',
    'extension': r'\bextension(?=[^a-zA-Z]|$)',
    # Employment/Education
    'occupation': r'\boccupation(?=[^a-zA-Z]|$)',
    'employer': r'\b(?:employer|employed\s+by)(?=[^a-zA-Z]|$)',
    'parent_employer': r'\bparent\s+employer(?=[^a-zA-Z]|$)',
    'patient_employer': r'\bpatient\s+employed\s+by(?=[^a-zA-Z]|$)',
    'student': r'\b(?:full\s+time\s+)?student(?=[^a-zA-Z]|$)',
    # ID fields
    'ssn': r'\b(?:ssn|soc\.?\s*sec\.?|social\s+security)(?=[^a-zA-Z]|$)',
    'drivers_license': r'\bdrivers?\s+license\s*#?(?=[^a-zA-Z]|$)',
    'member_id': r'\bmember\s+id(?=[^a-zA-Z]|$)',
    'policy_holder': r'\bpolicy\s+holder(?=[^a-zA-Z]|$)',
    # Address fields
    'address': r'\b(?:mailing\s+)?address(?=[^a-zA-Z]|$)',
    'city': r'\bcity(?=[^a-zA-Z]|$)',
    'state': r'\bstate(?=[^a-zA-Z]|$)',
    'zip': r'\bzip(?:\s+code)?(?=[^a-zA-Z]|$)',
    'apt': r'\bapt\s*#?(?=[^a-zA-Z]|$)',
    # Insurance fields
    'group_number': r'\b(?:group\s*#|plan\s*/\s*group\s+number)(?=[^a-zA-Z]|$)',
    'local_number': r'\blocal\s*#(?=[^a-zA-Z]|$)',
    'insurance_company': r'\b(?:insurance\s+company|name\s+of\s+insurance)(?=[^a-zA-Z]|$)',
    'dental_plan_name': r'\bdental\s+plan\s+name(?=[^a-zA-Z]|$)',
    'plan_group_number': r'\bplan\s*/\s*group\s+number(?=[^a-zA-Z]|$)',
    'insured_name': r"\b(?:name\s+of\s+)?insured(?:'?s)?\s+name(?=[^a-zA-Z]|$)",
    'relationship_to_insured': r'\b(?:patient\s+)?relationship\s+to\s+insured(?=[^a-zA-Z]|$)',
    'id_number': r'\bid\s+number(?=[^a-zA-Z]|$)',
    # Dental-specific fields
    'tooth_number': r'\btooth\s+(?:number|no\.?|#)(?=[^a-zA-Z]|$)',
    'physician_name': r'\bphysician\s+name(?=[^a-zA-Z]|$)',
    'dentist_name': r'\b(?:dentist|previous\s+dentist)\s+name(?=[^a-zA-Z]|$)',
    'dental_practice_name': r'\bname\s+of\s+(?:current|new|previous)?\s*dental\s+practice(?=[^a-zA-Z]|$)',
    'practice_name': r'\bpractice\s+name(?=[^a-zA-Z]|$)',
    # Misc
    'reason_for_visit': r'\breason\s+for\s+(?:today\'?s\s+)?visit(?=[^a-zA-Z]|$)',
    'previous_dentist': r'\bprevious\s+dentist(?=[^a-zA-Z]|$)',
    'date_of_release': r'\bdate\s+of\s+release(?=[^a-zA-Z]|$)',
}


def split_by_checkboxes_no_colon(line: str) -> List[str]:
    """
    Archivev12 Fix 1a: Split lines with checkboxes but no colons.
    Pattern: Label [ ] options ... Label [ ] options
    
    Strategy: Find field labels (capitalized words) that are followed by checkboxes,
    separated by 4+ spaces from the next field label.
    """
    # Find all potential field starts: Capital letter word followed by checkbox pattern
    # Look for pattern: 4+ spaces, then Capital letter starting a new field label
    # The key is to detect where one field ends and another begins by looking for spacing
    
    # Pattern to find split points: 4+ spaces followed by a capitalized label
    split_pattern = r'\s{4,}(?=[A-Z][A-Za-z\s]{2,30}?\s*\[)'
    
    split_positions = [m.start() for m in re.finditer(split_pattern, line)]
    
    if not split_positions:
        return [line]
    
    # Split at these positions
    segments = []
    last_pos = 0
    
    for pos in split_positions:
        segment = line[last_pos:pos].strip()
        if segment and re.search(CHECKBOX_ANY, segment):
            segments.append(segment)
        last_pos = pos
    
    # Add final segment
    final_segment = line[last_pos:].strip()
    if final_segment and re.search(CHECKBOX_ANY, final_segment):
        segments.append(final_segment)
    
    return segments if len(segments) >= 2 else [line]


def split_by_known_labels(line: str) -> List[str]:
    """
    Archivev12 Fix 1b + Production Improvement: Split lines based on known field labels.
    Handles: Work Phone (   )         Occupation
    Also handles: Are you a student? ... Mother's DOB ... Father's DOB
    NEW: Also handles adjacent labels with underscores (SSN_______ Date of Birth______)
    Enhanced: Now detects 2+ tabs as field separators (not just 4+ spaces)
    """
    # Find all known label matches in the line
    label_matches = []
    for field_key, pattern in KNOWN_FIELD_LABELS.items():
        for match in re.finditer(pattern, line, re.I):
            label_matches.append((match.start(), match.end(), field_key, match.group(0)))
    
    # Sort by position
    label_matches.sort()
    
    if len(label_matches) < 2:
        return [line]
    
    # Remove overlapping matches - keep the longer/more specific one
    filtered_matches = []
    for i, match in enumerate(label_matches):
        # Check if this match overlaps with any previous match
        overlaps = False
        for prev_match in filtered_matches:
            # Check if ranges overlap
            if not (match[1] <= prev_match[0] or match[0] >= prev_match[1]):
                # They overlap - keep the longer match
                overlaps = True
                break
        if not overlaps:
            filtered_matches.append(match)
    
    label_matches = filtered_matches
    
    if len(label_matches) < 2:
        return [line]
    
    # Production Improvement: Use more flexible splitting criteria
    segments = []
    last_added_idx = -1
    
    # If first label doesn't start at position 0, check for text before first big spacing gap
    # This handles cases like "Are you a student? Yes or No    Mother's DOB    Father's DOB"
    if label_matches[0][0] > 0:
        # Look for the first significant spacing gap (4+ spaces) before the second label
        if len(label_matches) >= 2:
            # Check text from start to second label
            text_before_second = line[:label_matches[1][0]]
            # Find position of first 4+ space gap
            gap_match = re.search(r'\s{4,}', text_before_second)
            if gap_match:
                # Extract everything before the gap as first segment
                before_gap = line[:gap_match.start()].strip()
                if before_gap and len(before_gap) > 5:
                    segments.append(before_gap)
    
    for i in range(len(label_matches)):
        start_pos = label_matches[i][0]
        
        if i + 1 < len(label_matches):
            # Check spacing between this label end and next label start
            end_this = label_matches[i][1]
            start_next = label_matches[i + 1][0]
            between = line[end_this:start_next]
            
            # Production Improvement: More flexible split criteria
            # Accept split if ANY of these conditions are met:
            # 1. 4+ consecutive spaces OR 1+ tabs OR mixed tab+spaces totaling 3+ (original criterion enhanced)
            # 2. Underscores/dashes followed by 1+ space and the next label (e.g., "______ Date of Birth")
            # 3. Multiple underscores/dashes/slashes between labels (indicating separate input fields)
            has_wide_spacing = bool(re.search(r'\s{4,}|\t+|[\t\s]{3,}', between))
            has_underscore_separator = bool(re.search(r'[_\-/]{3,}\s+', between))
            has_input_pattern = bool(re.search(r'[_\-]{3,}.*[_\-/()]{3,}', between))
            
            if not (has_wide_spacing or has_underscore_separator or has_input_pattern):
                continue
            
            # This is a valid split point
            end_pos = start_next
            segment = line[start_pos:end_pos].strip()
            if segment:
                segments.append(segment)
                last_added_idx = i
        else:
            # Last match - add from here to end
            segment = line[start_pos:].strip()
            if segment:
                segments.append(segment)
    
    # Only return segments if we got at least 2
    return segments if len(segments) >= 2 else [line]


def split_label_with_subfields(line: str) -> List[str]:
    """
    Archivev17 Fix: Handle pattern "Label: Sub1    Sub2    Sub3"
    Enhanced: Also handle patterns like "Label: Sub1 Sub2. Sub3" (period-separated)
    
    Detects lines where a single label (ending with colon) is followed by multiple
    sub-labels separated by wide spacing (4+ spaces) OR periods.
    
    Example: "Phone: Mobile                                  Home                                  Work"
    Example: "Phone: Mobile Home. Work"
    Should create: ["Mobile Phone", "Home Phone", "Work Phone"]
    
    Phase 4 Fix 8: Do NOT split lines that contain checkboxes - those are radio/dropdown fields
    with options, not separate input fields.
    """
    # Phase 4 Fix 8: Check if line has checkboxes first
    # Lines with checkboxes should be kept together as radio/dropdown fields
    if re.search(CHECKBOX_ANY, line):
        return [line]
    
    # Pattern 1: Label ending with colon, followed by multiple capitalized words separated by 4+ spaces
    match = re.match(r'^([A-Za-z][^:]{0,30}:)\s+([A-Z][a-z]+(?:\s{4,}[A-Z][a-z]+)+)\s*$', line.strip(), re.I)
    
    if match:
        main_label = match.group(1).strip(':').strip()  # e.g., "Phone"
        sublabels_text = match.group(2)  # e.g., "Mobile    Home    Work"
        
        # Split by 4+ spaces to get individual sub-labels
        sublabels = [s.strip() for s in re.split(r'\s{4,}', sublabels_text) if s.strip()]
    else:
        # Pattern 2: Label ending with colon, followed by multiple capitalized words separated by periods and/or spaces
        # Example: "Phone: Mobile Home. Work" or "Phone: Mobile Home Work"
        match = re.match(r'^([A-Za-z][^:]{0,30}:)\s+([A-Z][a-z]+(?:[\s.]+[A-Z][a-z]+)+)\s*$', line.strip(), re.I)
        
        if not match:
            return [line]
        
        main_label = match.group(1).strip(':').strip()  # e.g., "Phone"
        sublabels_text = match.group(2)  # e.g., "Mobile Home. Work"
        
        # Split by periods and/or spaces to get individual sub-labels
        # Remove empty strings and strip whitespace
        sublabels = [s.strip() for s in re.split(r'[\s.]+', sublabels_text) if s.strip()]
    
    # Must have at least 2 sub-labels to be valid
    if len(sublabels) < 2:
        return [line]
    
    # Create separate field entries
    # Format: "Sublabel Main-label" (e.g., "Mobile Phone", "Home Phone", "Work Phone")
    segments = []
    for sublabel in sublabels:
        # Create a natural-sounding field name
        if main_label.lower() in ['phone', 'number', 'address']:
            # For phone/number/address: "Mobile Phone", "Home Phone"
            segments.append(f"{sublabel} {main_label}")
        else:
            # For others: keep original format "Label: Sublabel"
            segments.append(f"{main_label}: {sublabel}")
    
    return segments if len(segments) >= 2 else [line]


def split_conditional_field_line(line: str) -> List[str]:
    """
    Archivev13 Fix: Handle conditional multi-field lines.
    Pattern: "Question? ... If condition: field1 ... field2"
    Example: "Are you a student? Yes or No If patient is a minor: Mother's DOB    Father's DOB"
    
    Note: Do NOT split yes/no questions with follow-up instructions like:
    "Are you taking medications? [ ] Yes [ ] No If yes, please explain:______"
    These should be handled by the compound yes/no logic instead.
    """
    # Look for "If ... :" pattern
    conditional_match = re.search(r'\b(if\s+[^:]{5,40}:)', line, re.I)
    if not conditional_match:
        return [line]
    
    # Check if this is a yes/no question with follow-up instructions
    # Pattern: checkbox or "Yes/No" before the "if" statement
    before_conditional = line[:conditional_match.start()]
    has_yesno_checkbox = bool(re.search(r'(?:yes|no|y|n)(?:\s*\[|\s*or\s)', before_conditional, re.I))
    
    # If it's "if yes" or "if no" followed by common follow-up keywords, don't split
    conditional_text = conditional_match.group(0).lower()
    is_followup_instruction = (
        ('if yes' in conditional_text or 'if no' in conditional_text) and
        any(keyword in conditional_text for keyword in ['explain', 'specify', 'list', 'describe', 'comment'])
    )
    
    if has_yesno_checkbox and is_followup_instruction:
        # This is a yes/no question with follow-up instructions, don't split
        return [line]
    
    conditional_text = conditional_match.group(0)
    conditional_start = conditional_match.start()
    
    # Split into:
    # 1. Text before conditional (should be a question)
    # 2. Text after conditional (contains multiple fields)
    before_conditional = line[:conditional_start].strip()
    after_conditional = line[conditional_match.end():].strip()
    
    segments = []
    
    # Add the question before conditional if it exists and is meaningful
    if before_conditional and len(before_conditional) > 5:
        segments.append(before_conditional)
    
    # Try to split the after_conditional part using known labels
    if after_conditional:
        after_segments = split_by_known_labels(after_conditional)
        if len(after_segments) > 1:
            segments.extend(after_segments)
        else:
            segments.append(after_conditional)
    
    return segments if len(segments) >= 2 else [line]


def split_compound_field_with_slashes(line: str) -> List[str]:
    """
    Enhancement 1: Split compound fields with slashes into separate fields.
    
    Examples:
        "Apt/Unit/Suite____" -> ["Apt", "Unit", "Suite"]
        "Name/Date/SSN____" -> ["Name", "Date", "SSN"]
        "Plan/Group Number____" -> ["Plan Number", "Group Number"]
    
    Only splits if:
    - Line contains field label(s) followed by slashes
    - Followed by underscores or other input markers (indicating it's a fillable field)
    - Each component is a reasonable field label (2+ characters)
    
    Returns:
        List of field segments (original line if no split needed)
    """
    # Pattern: One or more words/labels separated by slashes, followed by input markers
    # Example: "Apt/Unit/Suite________" or "Name/Date/SSN_______"
    pattern = r'^([A-Za-z][A-Za-z\s]*(?:/[A-Za-z][A-Za-z\s]*)+)\s*[:\-]?\s*([_\-\(\)]{3,})'
    match = re.match(pattern, line.strip())
    
    if not match:
        return [line]
    
    compound_label = match.group(1)
    input_marker = match.group(2)
    
    # Split by slash
    components = [c.strip() for c in compound_label.split('/') if c.strip()]
    
    # Only split if we have 2+ meaningful components
    if len(components) < 2:
        return [line]
    
    # Filter out single-letter components unless they're common abbreviations
    common_abbrevs = {'mi', 'm', 'f', 'n', 'y', 'apt', 'st', 'no'}
    valid_components = []
    for comp in components:
        # Keep if: 2+ chars, or is common abbreviation
        if len(comp) >= 2 or comp.lower() in common_abbrevs:
            valid_components.append(comp)
    
    if len(valid_components) < 2:
        return [line]
    
    # Create separate field lines
    segments = []
    for comp in valid_components:
        # Preserve some input markers for each field
        # Use proportional markers based on component count
        marker_len = max(3, len(input_marker) // len(valid_components))
        segments.append(f"{comp} {input_marker[:marker_len]}")
    
    return segments


def split_short_label_underscore_pattern(line: str) -> List[str]:
    """
    Category 1 Fix 1.1: Enhanced multi-field line splitting.
    
    Detects pattern: "Label1___ Label2___ Label3___" where labels are short words.
    Examples:
    - "First____ MI____ Last____"
    - "City____ State____ Zip____"
    - "Date____ Time____ Location____"
    
    This catches cases where the labels aren't in KNOWN_FIELD_LABELS.
    """
    # Pattern: Short word (2-15 chars, no spaces) followed by 3+ underscores
    # Must have at least 2 such patterns to be valid for splitting
    pattern = r'\b([A-Z][A-Za-z]{1,14})\s*_{3,}'
    matches = list(re.finditer(pattern, line))
    
    if len(matches) < 2:
        return [line]
    
    # Extract segments
    segments = []
    for i, match in enumerate(matches):
        if i + 1 < len(matches):
            # Extract from this label to the start of next label
            segment = line[match.start():matches[i+1].start()].strip()
            if segment:
                segments.append(segment)
        else:
            # Last segment - extract to end of line
            segment = line[match.start():].strip()
            if segment:
                segments.append(segment)
    
    # Only return segments if we successfully split into 2+ fields
    # and each segment has a reasonable length (not just underscores)
    if len(segments) >= 2 and all(len(s.strip('_').strip()) >= 2 for s in segments):
        return segments
    
    return [line]


def enhanced_split_multi_field_line(line: str) -> List[str]:
    """
    Archivev12 Fix 1 + Enhancement 1: Enhanced multi-field line splitting.
    Tries multiple strategies in order:
    0. Archivev15: Field label with inline checkbox (DON'T split these)
    0b. Phase 4 Fix 8: Preferred contact fields with checkboxes (DON'T split these)
    1. Enhancement 1: Compound fields with slashes (Apt/Unit/Suite)
    2. Archivev17: Label with sub-fields (Phone: Mobile Home Work)
    3. Conditional field patterns (Archivev13)
    4. Existing pattern (colon + checkbox)
    5. Checkboxes without colons
    6. Known label patterns with spacing
    7. Category 1 Fix 1.1: Short label + underscore pattern (NEW)
    """
    # Archivev15 Fix 1: Check if this is a field with inline checkbox option
    # These should NOT be split, so return early
    if detect_field_with_inline_checkbox(line):
        return [line]
    
    # Phase 4 Fix 8: Check if this is a preferred contact field with checkboxes
    # These should NOT be split by known label patterns (Home Phone, Work Phone, etc.)
    # as those are options for a single radio field, not separate input fields
    has_preferred_contact = any(p.search(line) for p in PREFERRED_CONTACT_PATTERNS)
    if has_preferred_contact and re.search(CHECKBOX_ANY, line):
        return [line]
    
    # Enhancement 1: Try compound field splitting with slashes
    result = split_compound_field_with_slashes(line)
    if len(result) > 1:
        return result
    
    # Archivev17 Fix: Check if this is "Label: Sub1  Sub2  Sub3" pattern
    result = split_label_with_subfields(line)
    if len(result) > 1:
        return result
    
    # Try conditional field pattern first (Archivev13)
    result = split_conditional_field_line(line)
    if len(result) > 1:
        return result
    
    # Try existing pattern (backward compatible)
    result = split_multi_question_line(line)
    if len(result) > 1:
        return result
    
    # Try checkboxes without colons
    result = split_by_checkboxes_no_colon(line)
    if len(result) > 1:
        return result
    
    # Try known label detection
    result = split_by_known_labels(line)
    if len(result) > 1:
        return result
    
    # Category 1 Fix 1.1: Try short label + underscore pattern (NEW)
    result = split_short_label_underscore_pattern(line)
    if len(result) > 1:
        return result
    
    return [line]


def detect_sex_gender_field(line: str) -> Optional[Tuple[str, str]]:
    """
    Archivev12 Fix 2a: Detect sex/gender field patterns.
    Phase 4 Fix 1: Enhanced to handle checkbox patterns like "Sex □ Male □ Female"
    Returns: (field_type, segment) or None
    
    For lines with checkboxes (e.g., "Sex [ ] Male [ ] Female"), extract the entire field.
    For lines with text options (e.g., "Sex M or F"), extract up to the pattern.
    """
    for pattern in SEX_GENDER_PATTERNS:
        match = pattern.search(line)
        if match:
            # Check if this is a checkbox-based field
            # Look for checkboxes after the label
            after_label = line[match.end():]
            has_checkboxes = bool(re.search(CHECKBOX_ANY, after_label))
            
            if has_checkboxes or match.group(0).count('□') >= 2 or match.group(0).count('[') >= 2:
                # This is a checkbox-based field, extract the entire field including all checkboxes
                # Look for significant spacing (4+ spaces) followed by a capital letter (indicating next field)
                # OR look for "Marital Status" which commonly follows on same line
                next_field_match = re.search(r'\s{4,}[A-Z]', line[match.start():])
                if next_field_match:
                    # Extract up to next field
                    end_pos = match.start() + next_field_match.start()
                    return ("sex_gender", line[:end_pos].strip())
                else:
                    # No clear next field boundary, extract to end
                    # But check if line also contains "Marital Status" - if so, split there
                    marital_match = re.search(r'\bmarital\s+status', line[match.end():], re.I)
                    if marital_match:
                        # Extract up to "Marital Status"
                        end_pos = match.end() + marital_match.start()
                        return ("sex_gender", line[:end_pos].strip())
                    else:
                        return ("sex_gender", line.strip())
            else:
                # Text-based options (M or F), extract up to match end
                return ("sex_gender", line[:match.end()].strip())
    return None


def detect_marital_status_field(line: str) -> Optional[Tuple[str, str]]:
    """
    Archivev12 Fix 2b: Detect marital status field patterns.
    Phase 4 Fix 2: Enhanced to handle checkbox patterns
    Returns: (field_type, extracted_segment) or None
    """
    for pattern in MARITAL_STATUS_PATTERNS:
        match = pattern.search(line)
        if match:
            # For "Please Circle One:" pattern, need to capture options after it
            if "circle" in match.group(0).lower():
                # Extract from "Please Circle One:" onwards
                segment = line[match.start():].strip()
                return ("marital_status", segment)
            else:
                # For checkbox-based patterns, extract to end of line
                # (assume marital status is last field on line)
                segment = line[match.start():].strip()
                return ("marital_status", segment)
    return None


def detect_preferred_contact_field(line: str) -> Optional[Tuple[str, str]]:
    """
    Phase 4 Fix 3/8: Detect preferred method of contact field patterns.
    Returns: (field_type, extracted_segment) or None
    
    This field is typically a standalone line with a question followed by multiple checkbox options.
    Example: "What is your preferred method of contact? □ Mobile Phone □ Home Phone □ Work Phone □ E-mail"
    """
    for pattern in PREFERRED_CONTACT_PATTERNS:
        match = pattern.search(line)
        if match:
            # Extract the complete segment (question + all options)
            # Since this is typically a standalone field, extract entire line
            segment = line[match.start():].strip()
            
            # Verify we have multiple checkbox options (should have at least 2)
            checkbox_count = len(re.findall(CHECKBOX_ANY, segment))
            if checkbox_count >= 2:
                return ("preferred_contact", segment)
    return None


def split_complex_multi_field_line(line: str) -> List[str]:
    """
    Archivev12 Fix 2c: Handle complex lines with multiple fields of different types.
    E.g., "Sex M or F   Soc. Sec. #   Please Circle One: Single Married..."
    """
    segments = []
    remaining = line
    
    # Phase 4 Fix: Try to detect and extract preferred contact field first (it's usually a complete standalone line)
    pref_result = detect_preferred_contact_field(line)
    if pref_result:
        field_type, segment = pref_result
        segments.append(segment)
        return segments  # Preferred contact lines are typically standalone, return early
    
    # Try to detect and extract sex/gender field
    sex_result = detect_sex_gender_field(line)
    if sex_result:
        field_type, segment = sex_result
        segments.append(segment)
        # Remove this segment from remaining, accounting for position
        sex_pos = line.find(segment)
        if sex_pos == 0:
            remaining = line[len(segment):].strip()
        else:
            remaining = line[sex_pos + len(segment):].strip()
    
    # Try to detect and extract marital status field from remaining
    marital_result = detect_marital_status_field(remaining)
    if marital_result:
        field_type, segment = marital_result
        # Find the position of marital status segment in remaining
        marital_start = remaining.lower().find("please circle") if "please circle" in remaining.lower() else remaining.lower().find("marital")
        
        if marital_start > 0:
            # There's content before marital status (like SSN)
            before = remaining[:marital_start].strip()
            if before and len(before) > 3:  # Has significant content before
                segments.append(before)
        
        # Add the marital status field
        segments.append(segment)
        remaining = ""
    elif remaining:
        # If we extracted sex/gender, add remaining as separate field
        if segments and len(remaining) > 3:
            segments.append(remaining)
    
    return segments if len(segments) >= 2 else [line]


def preprocess_lines(lines: List[str]) -> List[str]:
    """
    Preprocess lines before main parsing.
    Currently handles: splitting multi-question lines with enhanced strategies.
    Enhancement: Recursively processes split segments to handle compound fields.
    
    Parity Improvements:
    - Improvement #1: Detect and split colon-delimited multi-field blocks
    - Improvement #2: Enhanced multi-sub-field label splitting
    - Improvement #1: Field label separation from blanks
    - Improvement #5: Compound field line normalization
    """
    processed = []
    for line in lines:
        # Skip empty lines
        if not line.strip():
            processed.append(line)
            continue
        
        # PARITY FIX: Check for tab-separated signature lines FIRST
        # These should NOT be split by colon-delimited parsing
        # Example: "Patient's name (please print)\tSignature\tDate"
        if is_tabulated_signature_line(line):
            # Preserve the line as-is for signature parsing in main loop
            processed.append(line)
            continue
        
        # Improvement #1: Separate field labels from underscores
        line = separate_field_label_from_blanks(line)
        
        # Improvement #5: Try to normalize compound field lines
        compound_fields = normalize_compound_field_line(line)
        if len(compound_fields) > 1:
            # Successfully split compound fields
            processed.extend(compound_fields)
            continue
        
        # PARITY FIX: Skip colon splitting for consent paragraphs
        # Long lines with consent/legal text should not be split into fields
        # Example: "Breakage: Due to the types of materials..." is a paragraph, not a field
        if is_consent_paragraph(line) or len(line) > 200:
            processed.append(line)
            continue
        
        # NEW Improvement #1: Try colon-delimited field splitting
        # This handles insurance/registration blocks like "Name: State: Holder: Birth Date:"
        if should_split_line_into_fields(line):
            colon_split = split_colon_delimited_fields(line)
            if colon_split:
                # Successfully split into multiple fields
                # For each field, check if the value area contains sub-fields to extract
                for field_dict in colon_split:
                    title = field_dict['title']
                    value_area = field_dict.get('value_area', '')
                    
                    # Try to detect sub-fields in the value area
                    # After separate_field_label_from_blanks, format is "First: ___ MI: ___ Last: ___"
                    # So check for multiple "Label: ___" patterns in the value area
                    if value_area and len(value_area) > 10:
                        # Check for colon-delimited sub-fields in value area
                        # Pattern: "Label: ___" appearing multiple times
                        subfield_pattern = r'\b([A-Z][A-Za-z\s]{1,20}):\s*___'
                        subfield_matches = list(re.finditer(subfield_pattern, value_area))
                        
                        if len(subfield_matches) >= 2:
                            # Extract each sub-field
                            for match in subfield_matches:
                                subfield_label = match.group(1).strip()
                                # Create compound title: "Patient Name - First"
                                compound_title = f"{title} - {subfield_label}"
                                processed.append(f"{compound_title}: ___")
                            continue
                    
                    # No sub-fields found, create simple field
                    field_line = f"{title}: ___"
                    processed.append(field_line)
                continue
        
        # NEW Improvement #2: Try multi-sub-field splitting
        # Handles "Phone: Mobile ___ Home ___ Work ___"
        subfield_split = split_multi_subfield_line(line)
        if subfield_split:
            for field_dict in subfield_split:
                field_line = f"{field_dict['title']}: ___"
                processed.append(field_line)
            continue
        
        # Archivev12 Fix: Try multiple splitting strategies
        # Strategy 1: Check if line has sex/gender, marital, or preferred contact patterns
        has_sex_gender = any(p.search(line) for p in SEX_GENDER_PATTERNS)
        has_marital = any(p.search(line) for p in MARITAL_STATUS_PATTERNS)
        has_preferred_contact = any(p.search(line) for p in PREFERRED_CONTACT_PATTERNS)
        
        if has_sex_gender or has_marital or has_preferred_contact:
            # Try complex multi-field detection first for these special cases
            split_lines = split_complex_multi_field_line(line)
            if len(split_lines) == 1:
                # If complex didn't split, try enhanced
                split_lines = enhanced_split_multi_field_line(line)
        else:
            # Strategy 2: Enhanced multi-field splitting for regular cases
            split_lines = enhanced_split_multi_field_line(line)
        
        # Enhancement: Recursively process split segments
        # This catches compound fields like "Apt/Unit/Suite" that were created from splitting
        if len(split_lines) > 1:
            for segment in split_lines:
                # Try to split each segment further (e.g., compound fields with slashes)
                # But avoid infinite recursion by only trying compound splitting
                compound_split = split_compound_field_with_slashes(segment)
                processed.extend(compound_split)
        else:
            processed.extend(split_lines)
    
    return processed

# ---------- Options & typing

def extract_orphaned_checkboxes_and_labels(current_line: str, next_line: str) -> List[Tuple[str, Optional[bool]]]:
    """
    Fix 2: When current line has checkboxes but minimal text,
    and next line has label text, associate them by column position.
    
    Example:
      Current: "[ ]           [ ]           [ ]"
      Next:    "Anemia        Diabetes      Cancer"
      Returns: [("Anemia", None), ("Diabetes", None), ("Cancer", None)]
    """
    # Count checkboxes on current line
    checkbox_matches = list(re.finditer(CHECKBOX_ANY, current_line))
    if len(checkbox_matches) < 2:
        return []
    
    # Check if current line has minimal text (mostly just checkboxes)
    text_after_boxes = re.sub(CHECKBOX_ANY, '', current_line).strip()
    if len(text_after_boxes) > 50:  # Has substantial text, not orphaned
        return []
    
    # Check if next line has no checkboxes but has text
    if re.search(CHECKBOX_ANY, next_line):
        return []  # Next line also has checkboxes, not the label line
    
    next_stripped = next_line.strip()
    if not next_stripped or len(next_stripped) < 3:
        return []
    
    # Split next line into words/phrases by whitespace
    # Assume labels align roughly with checkbox positions
    words = re.split(r'\s{2,}', next_stripped)  # Split on 2+ spaces
    words = [w.strip() for w in words if w.strip()]
    
    # Match count: ideally num_words == num_checkboxes
    num_boxes = len(checkbox_matches)
    if len(words) < 2 or len(words) > num_boxes + 2:  # Some tolerance
        return []
    
    # Associate words with checkboxes
    options = []
    for word in words[:num_boxes]:  # Take up to num_boxes words
        if len(word) > 1:
            options.append((word, None))
    
    return options if len(options) >= 2 else []

def extract_title_from_inline_checkboxes(line: str) -> str:
    """
    Extract the question/prompt text before the first checkbox marker (Fix 1).
    
    Example:
        "How did you hear? [ ] Google [ ] Friend" -> "How did you hear?"
        "Gender: [ ] Male [ ] Female" -> "Gender:"
    """
    # Pattern to match text before first checkbox
    match = re.match(r'^(.*?)(?:\[\s*\]|\[x\]|☐|☑|□|■|❒|◻)', line)
    if match:
        title = match.group(1).strip()
        # Clean up trailing colons, question marks, etc
        title = title.rstrip(':? ').strip()
        if title:
            return title
    # Fallback: return cleaned line
    return re.sub(CHECKBOX_ANY, '', line).strip()

# clean_field_title moved to modules/question_parser.py (Patch 7 Phase 1)

# clean_token moved to modules/question_parser.py (Patch 7 Phase 1)

# normalize_opt_name moved to modules/question_parser.py (Patch 7 Phase 1)

def option_from_bullet_line(ln: str) -> Optional[Tuple[str, Optional[bool]]]:
    s = ln.strip()
    if not BULLET_RE.match(s):
        return None
    # Check if there are multiple checkboxes on this line (grid format, not bullet)
    checkbox_count = len(re.findall(CHECKBOX_ANY, s))
    if checkbox_count > 1:
        return None  # This is inline grid format, not a bullet
    s = CHECKBOX_MARK_RE.sub("", s, count=1)
    s = re.sub(r"^\s*[-*•·]\s+", "", s)
    name = clean_token(s)
    if not name:
        return None
    low = name.lower()
    if low in ("yes", "y"): return ("Yes", True)
    if low in ("no", "n"):  return ("No", False)
    return (name, None)

def detect_field_with_inline_checkbox(line: str) -> Optional[Tuple[str, str]]:
    """
    Archivev15 Fix 1: Detect lines with field label followed by inline checkbox option.
    
    Pattern: "Field Label:    <spaces>    [ ] Option text"
    
    Returns:
        Tuple of (field_label, checkbox_option) if pattern detected, None otherwise
    
    Examples:
        "Cell Phone:                         [ ] Yes, send me Text Message alerts"
        -> ("Cell Phone", "Yes, send me Text Message alerts")
        
        "E-mail Address:                     [ ] Yes, send me alerts via Email"
        -> ("E-mail Address", "Yes, send me alerts via Email")
    """
    # Look for pattern: text ending with colon, followed by spaces, then checkbox and text
    pattern = r'^(.+?):\s{5,}' + CHECKBOX_ANY + r'\s+(.+)$'
    match = re.match(pattern, line.strip())
    
    if match:
        field_label = match.group(1).strip()
        checkbox_text = match.group(2).strip()
        
        # Validate: field label should be reasonably short (not a question)
        # and checkbox text should be meaningful
        if len(field_label) <= 50 and len(checkbox_text) >= 5:
            return (field_label, checkbox_text)
    
    return None


def options_from_inline_line(ln: str) -> List[Tuple[str, Optional[bool]]]:
    """
    Enhanced to handle grid/multi-column layouts (Fix 1).
    Splits checkboxes that appear in columns with significant spacing.
    """
    s_norm = normalize_glyphs_line(ln)
    
    # First try: existing inline choice regex
    items: List[Tuple[str, Optional[bool]]] = []
    for m in INLINE_CHOICE_RE.finditer(s_norm):
        raw_label = m.group(1)
        
        # Fix 3: Split on excessive spacing BEFORE clean_token (which collapses spaces)
        parts = re.split(r'\s{5,}', raw_label)
        if len(parts) > 1:
            # Take only first part (the actual checkbox label)
            original = raw_label
            raw_label = next((p.strip() for p in parts if p.strip()), raw_label)
            # DEBUG
            if 'Bad Bite' in original and 'Please list' in original:
                import sys
                print(f"DEBUG: Split '{original[:60]}...' -> '{raw_label}'", file=sys.stderr)
        
        label = clean_token(raw_label)
        if not label: continue
        low = label.lower()
        if low in ("yes", "y"): items.append(("Yes", True))
        elif low in ("no", "n"): items.append(("No", False))
        else: items.append((label, None))
    
    # If we found options with existing method, use it (unless it looks like grid layout)
    if items and len(items) < 3:
        return items
    
    # NEW: Grid detection - look for multiple checkboxes with wide spacing (Fix 1)
    checkbox_positions = []
    for m in re.finditer(CHECKBOX_ANY, s_norm):
        checkbox_positions.append(m.start())
    
    if len(checkbox_positions) >= 3:  # Multiple checkboxes suggest grid
        # Split line into segments based on checkbox positions
        options = []
        for i, start_pos in enumerate(checkbox_positions):
            # Extract text after this checkbox until next checkbox or EOL
            if i + 1 < len(checkbox_positions):
                end_pos = checkbox_positions[i + 1]
            else:
                end_pos = len(s_norm)
            
            segment = s_norm[start_pos:end_pos]
            # Remove checkbox token and extract label
            label = re.sub(CHECKBOX_ANY, '', segment).strip()
            
            # Fix 3: Better cleaning for grid layouts
            # 1. Split on excessive spacing (5+ spaces = likely column boundary)
            parts = re.split(r'\s{5,}', label)
            if len(parts) > 1:
                # Take the first non-empty part (the actual label)
                label = next((p.strip() for p in parts if p.strip()), label)
            
            # 2. Split on category headers that appear mid-text (Fix 3 enhancement)
            # Pattern: text followed by category name followed by more text
            category_pattern = r'\b(Cardiovascular|Gastrointestinal|Neurological|Viral|Hematologic|Lymphatic|Infections?)\b'
            # Check if category appears in middle of text
            match = re.search(category_pattern, label, re.I)
            if match:
                # Split before the category and take the first part
                label = label[:match.start()].strip()
            
            # 3. Handle merged medical terms (Fix 3 - complex case)
            # Pattern: "Word1 Word2 (parenthetical) Word3" where Word3 looks unrelated
            # Example: "Artificial Angina (chest pain) Valve" should become "Artificial Heart Valve" or just "Artificial Valve"
            # If we have pattern like "X Y (...) Z" where Y and Z are both medical terms, keep only first part
            paren_match = re.search(r'^(.+?)\s+\w+\s*\([^)]+\)\s+(\w+)$', label)
            if paren_match:
                # This looks like merged terms. Try to clean it up.
                first_part = paren_match.group(1).strip()
                last_word = paren_match.group(2).strip()
                # If first part is short (1-2 words), combine with last word
                if len(first_part.split()) <= 2:
                    label = f"{first_part} {last_word}"
                else:
                    # Keep first part only
                    label = first_part
            
            # 4. Collapse remaining multiple spaces
            label = re.sub(r'\s{2,}', ' ', label)
            
            # 5. Remove trailing checkbox artifacts
            label = label.strip('[]')
            
            # 6. Filter out standalone category headers
            category_headers = r'^(Type|Cardiovascular|Gastrointestinal|Neurological|Viral|Women|Hematologic|Lymphatic|Infections?|Additional)$'
            if re.match(category_headers, label.strip(), re.I):
                continue
            
            # 7. Apply standard token cleaning
            label = clean_token(label)
            
            if label and len(label) > 1 and label.lower() not in YESNO_SET:
                options.append((label, None))
        
        if len(options) >= 2:  # Changed from >= 3 to be more permissive (Fix 3)
            return options
    
    return items  # Fall back to existing method

# classify_input_type moved to modules/question_parser.py (Patch 7 Phase 1)

# classify_date_input moved to modules/question_parser.py (Patch 7 Phase 1)

# ---------- Model

@dataclass
class Question:
    key: str
    title: str
    section: str
    type: str
    optional: bool = False
    control: Dict = field(default_factory=dict)
    conditional_on: Optional[List[Tuple[str, str]]] = None

# slugify moved to modules/question_parser.py (Patch 7 Phase 1)

# ---------- Archivev8 Fix 4: Clean Malformed Option Text

# clean_option_text moved to modules/question_parser.py (Patch 7 Phase 1)

# make_option moved to modules/question_parser.py (Patch 7 Phase 1)

# ---------- Composite / multi-label fan-out

CANON_LABELS = {
    "last name": "Last Name",
    "first name": "First Name",
    "mi": "MI",
    "middle initial": "MI",
    "nickname": "Preferred Name",
    "preferred name": "Preferred Name",
    "patient name": "Patient Name",
    "date of birth": "Date of Birth",
    "dob": "Date of Birth",
    "age": "Age",
    "address": "Address",
    "mailing address": "Address",
    "city": "City",
    "state": "State",
    "zip": "Zipcode",
    "zipcode": "Zipcode",
    "email": "Email",
    "e mail": "Email",
    "phone": "Phone",
    "home phone": "Home Phone",
    "mobile phone": "Mobile Phone",
    "cell phone": "Mobile Phone",
    "work phone": "Work Phone",
    "ssn": "SSN",
    "social security": "SSN",
    "soc sec": "SSN",
    "ss#": "SSN",
    "employer": "Employer",
    "employer (if different from above)": "Employer",
    "occupation": "Occupation",
    "emergency contact": "Emergency Contact",
    "emergency name": "Emergency Contact Name",
    "emergency phone": "Emergency Phone",
    "emergency relationship": "Emergency Relationship",
    "relationship": "Relationship",
    "person responsible for account": "Responsible Party Name",
    "responsible party name": "Responsible Party Name",
    "responsible party": "Responsible Party Name",
    "drivers license #": "Driver's License #",
    "driver's license #": "Driver's License #",
    "drivers license": "Driver's License #",
    "driver license": "Driver's License #",
    "name of parent": "Parent Name",
    "parent name": "Parent Name",
    "parent soc. sec. #": "Parent SSN",
    "parent ssn": "Parent SSN",
    # Insurance
    "insureds name": "Insured's Name",
    "insured name": "Insured's Name",
    "insureds": "Insured's Name",
    "insured": "Insured's Name",
    "subscriber name": "Insured's Name",
    "member name": "Insured's Name",
    "policy holder": "Insured's Name",
    "relationship to insured": "Relationship to Insured",
    "relationship to subscriber": "Relationship to Insured",
    "relationship to policy holder": "Relationship to Insured",
    "relationship to member": "Relationship to Insured",
    "id number": "ID Number",
    "id no": "ID Number",
    "member id": "ID Number",
    "policy id": "ID Number",
    "identification number": "ID Number",
    "group number": "Group Number",
    "group #": "Group Number",
    "grp #": "Group Number",
    "plan/group number": "Group Number",
    "insurance company": "Insurance Company",
    "insurance phone": "Insurance Phone",
    "insurance phone #": "Insurance Phone",
    "customer service phone": "Insurance Phone",
    "cust svc phone": "Insurance Phone",
    "address on card": "Insurance Address",
    "insurance address": "Insurance Address",
}

LABEL_ALTS = sorted(CANON_LABELS.keys(), key=len, reverse=True)

def _sanitize_words(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def try_split_known_labels(line: str) -> List[str]:
    """
    Detect and split compound field labels from a single line.
    
    Examples of compound fields that should be split:
    - "Name of insured / Birthdate / SSN"
    - "First Name, Last Name, Date of Birth"
    - "Patient Name | DOB | Phone"
    
    This addresses the Modento schema requirement that each question key be unique.
    """
    s_raw = normalize_glyphs_line(line)
    s = collapse_spaced_caps(s_raw).strip()
    if not s or len(s) > 220 or s.endswith("."):
        return []
    
    # collapse repeated phrases like "Insured's Name Insured's Name"
    s_de_rep = re.sub(r"(\binsured'?s?\s+name\b)(\s+\1)+", r"\1", s, flags=re.I)
    
    # Enhanced: Check for explicit compound field separators (/, |, comma followed by capital letter)
    # These are strong indicators that the line contains multiple distinct fields
    has_explicit_separators = bool(re.search(r'\s+[/|]\s+|\s+/\s+|\|\s+', s_de_rep))
    
    # Also detect comma-separated fields (e.g., "Last Name, First Name, DOB")
    # Only if commas are followed by capital letters (field names typically start with caps)
    has_comma_fields = bool(re.search(r',\s+[A-Z]', s_de_rep))
    
    s_sanit = _sanitize_words(s_de_rep)
    hits: List[Tuple[int, str]] = []
    for phrase in LABEL_ALTS:
        p = _sanitize_words(phrase)
        # Archivev20 Fix 9: Use word boundary matching to avoid false positives
        # "message alerts" was matching "age" and "ss", causing incorrect label extraction
        pattern = r'\b' + re.escape(p) + r'\b'
        match = re.search(pattern, s_sanit)
        if match:
            hits.append((match.start(), CANON_LABELS.get(phrase, phrase.title())))
    
    # Enhanced: If we have explicit separators and found at least 2 distinct fields, split them
    # This catches compound fields like "Name / DOB / SSN" even if only 2 of 3 are in CANON_LABELS
    if len(hits) < 2:
        # If we have explicit separators, try to extract field names even if not in CANON_LABELS
        if has_explicit_separators or has_comma_fields:
            # Split by separators and try to identify field names
            parts = re.split(r'\s*[/|,]\s*', s_de_rep)
            if len(parts) >= 2:
                # Only return if we can identify at least 2 fields that look like field labels
                # (start with capital letter, contain key field terms)
                potential_fields = []
                for part in parts:
                    part = part.strip()
                    # Check if this looks like a field label (not just a value)
                    if part and (
                        len(part) >= 3 and 
                        part[0].isupper() and
                        not part.replace(' ', '').isdigit()  # Not just numbers
                    ):
                        potential_fields.append(part)
                
                if len(potential_fields) >= 2:
                    # Return the canonicalized versions from hits if available, otherwise the raw field names
                    result = []
                    for part in potential_fields:
                        # Check if this field is in our hits
                        matched = False
                        part_sanit = _sanitize_words(part)
                        for phrase in LABEL_ALTS:
                            if _sanitize_words(phrase) in part_sanit:
                                canon = CANON_LABELS.get(phrase, phrase.title())
                                if canon not in result:
                                    result.append(canon)
                                    matched = True
                                    break
                        # If not matched to CANON_LABELS, use the raw part as field name
                        if not matched and part not in result:
                            result.append(part.strip())
                    
                    if len(result) >= 2:
                        return result
        
        return []
    
    hits.sort(key=lambda t: t[0])
    labels = []
    for _pos, canon in hits:
        if canon not in labels:
            labels.append(canon)
    return labels


# Grid parsing functions now imported from modules.grid_parser
# These functions handle detection and parsing of multi-column checkbox grids:
# - looks_like_grid_header, detect_table_layout, parse_table_to_questions
# - chunk_by_columns, detect_column_boundaries
# - detect_multicolumn_checkbox_grid, parse_multicolumn_checkbox_grid
# - extract_text_for_checkbox, extract_text_only_items_at_columns

# ---------- Compound Yes/No

COMPOUND_YN_RE = re.compile(
    rf"(?P<prompt>[^[]*?)\s*{CHECKBOX_ANY}\s*(?:Yes|Y)\s*{CHECKBOX_ANY}\s*(?:No|N)\b",
    re.I,
)

def extract_compound_yn_prompts(line: str) -> List[str]:
    prompts = []
    for m in COMPOUND_YN_RE.finditer(normalize_glyphs_line(line)):
        p = collapse_spaced_caps(m.group("prompt")).strip(" :;-")
        if p:
            # Archivev19 Fix 2: Check if there's continuation text after the Yes/No checkboxes
            # Example: "...Actonel/ [ ] Yes [ ] No other medications containing bisphosphonates?"
            # We want to include the continuation text in the prompt
            match_end = m.end()
            remaining_text = line[match_end:].strip()
            
            # If there's text after Yes/No and it starts with lowercase or connecting words,
            # it's likely a continuation of the question
            if remaining_text and (
                re.match(r'^[a-z(]', remaining_text) or 
                re.match(r'^(and|or|if|but|then|with|of|for|to|other|additional)\b', remaining_text, re.I)
            ):
                # Remove any leading "and" or similar connectors that might cause awkwardness
                continuation = remaining_text
                # Append continuation to prompt
                p = p + " " + continuation
            
            prompts.append(p)
    if not prompts:
        m2 = YN_SIMPLE_RE.search(line)
        if m2:
            p = collapse_spaced_caps(m2.group("prompt")).strip(" :;-")
            if p: prompts.append(p)
    seen = set(); out=[]
    for p in prompts:
        if p not in seen:
            out.append(p); seen.add(p)
    return out

# ---------- Archivev8 Fix 1: Orphaned Checkbox Labels Detection

def has_orphaned_checkboxes(line: str) -> bool:
    """
    Detect if a line has multiple checkboxes but very little text (orphaned checkboxes).
    
    Example: "[ ]                       [ ]                               [ ]"
    
    Returns:
        True if checkboxes appear orphaned (labels likely on next line)
    """
    checkbox_count = len(re.findall(CHECKBOX_ANY, line))
    if checkbox_count < 2:
        return False
    
    # Remove checkboxes and see how much text remains
    text_without_checkboxes = re.sub(CHECKBOX_ANY, '', line).strip()
    
    # Split by whitespace to count words
    words = [w for w in text_without_checkboxes.split() if w.strip()]
    
    # Heuristic: If we have many checkboxes but very few words, labels are likely orphaned
    # Allow 1-2 short words per checkbox (like "Sickle Cell Disease" at the end)
    # But if most checkboxes have no adjacent text, they're orphaned
    if checkbox_count >= 3 and len(words) <= 2:
        return True
    
    # Alternative check: very sparse text density
    if len(text_without_checkboxes) < (checkbox_count * 5):
        return True
    
    return False


def extract_orphaned_labels(label_line: str) -> List[str]:
    """
    Extract labels from a line that appears to contain orphaned labels.
    
    A label line may have some checkboxes at the end, but should have
    labels without checkboxes at the beginning/middle.
    
    Returns:
        List of label strings
    """
    stripped = label_line.strip()
    if not stripped:
        return []
    
    # Split by 3+ spaces to get individual labels
    # This is a common pattern in grid layouts
    parts = re.split(r'\s{3,}', stripped)
    
    # Clean and filter labels
    cleaned_labels = []
    for part in parts:
        part = part.strip()
        # Skip if this part starts with a checkbox (it's properly paired)
        if re.match(CHECKBOX_ANY, part):
            # This part has a checkbox, extract the label after it
            label = re.sub(CHECKBOX_ANY, '', part).strip()
            if label and len(label) >= 2 and not label.isdigit():
                cleaned_labels.append(label)
        else:
            # This part has no checkbox - it's an orphaned label
            if len(part) >= 2 and not part.isdigit():
                cleaned_labels.append(part)
    
    return cleaned_labels


def associate_orphaned_labels_with_checkboxes(
    checkbox_line: str,
    label_line: str
) -> List[Tuple[str, Optional[bool]]]:
    """
    Associate orphaned labels with checkboxes based on occurrence order.
    
    Args:
        checkbox_line: Line with checkboxes but minimal text
        label_line: Next line with labels (may have some checkboxes at end)
    
    Returns:
        List of (label, checked_state) tuples
    """
    # Check if this actually looks like orphaned pattern
    if not has_orphaned_checkboxes(checkbox_line):
        return []
    
    labels = extract_orphaned_labels(label_line)
    if not labels:
        return []
    
    # Count checkboxes in the checkbox line
    checkbox_matches = list(re.finditer(CHECKBOX_ANY, checkbox_line))
    num_checkboxes = len(checkbox_matches)
    
    if num_checkboxes == 0:
        return []
    
    # Also check if checkbox line has any labels at the end
    # e.g., "[ ]  [ ]  [ ]  [ ]  [ ] Sickle Cell Disease"
    text_after_last_checkbox = checkbox_line[checkbox_matches[-1].end():].strip()
    checkbox_line_labels = []
    if text_after_last_checkbox and len(text_after_last_checkbox) > 3:
        # There's a label on the checkbox line itself
        checkbox_line_labels.append(text_after_last_checkbox)
    
    # If we have orphaned labels and checkboxes, associate them
    options = []
    
    # Add all labels from the label line
    for label in labels:
        options.append((label, None))
    
    # Then add any label from the checkbox line itself
    for label in checkbox_line_labels:
        options.append((label, None))
    
    return options

# ---------- Fix 2: Enhanced "If Yes" Detection

def extract_yn_with_followup(line: str) -> Tuple[Optional[str], bool]:
    """
    Extract Yes/No question and determine if it has a follow-up.
    
    Returns:
        (question_text, has_followup)
    
    Examples:
        "Are you pregnant? [ ] Yes [ ] No If yes, please explain"
        -> ("Are you pregnant?", True)
        
        "Do you smoke? [ ] Yes [ ] No"
        -> ("Do you smoke?", False)
    """
    # Try explicit "if yes" pattern first
    match = IF_YES_FOLLOWUP_RE.search(line)
    if match:
        question = match.group(1).strip()
        return (question, True)
    
    # Try inline "if yes" pattern
    match = IF_YES_INLINE_RE.search(line)
    if match:
        question = match.group(1).strip()
        return (question, True)
    
    # Try existing compound pattern
    prompts = extract_compound_yn_prompts(line)
    if prompts:
        # Check if line mentions "if yes" anywhere
        has_followup = bool(re.search(r'\bif\s+yes\b', line, re.I))
        return (prompts[0], has_followup)
    
    return (None, False)


def create_yn_question_with_followup(
    question_text: str,
    section: str,
    key_base: Optional[str] = None
) -> List['Question']:
    """
    Create a Yes/No radio question with a conditional follow-up input field.
    
    Args:
        question_text: The question text
        section: Current section
        key_base: Base key (if None, generated from question_text)
    
    Returns:
        List of 2 Questions: [radio_question, followup_input]
    """
    if not key_base:
        key_base = slugify(question_text)
    
    # Main Yes/No question
    main_q = Question(
        key_base,
        question_text,
        section,
        "radio",
        control={"options": [make_option("Yes", True), make_option("No", False)]}
    )
    
    # Follow-up input field (conditional on Yes)
    followup_key = f"{key_base}_explanation"
    followup_q = Question(
        followup_key,
        "Please explain",
        section,
        "input",
        control={"input_type": "text", "hint": "Please provide details"}
    )
    followup_q.conditional_on = [(key_base, "yes")]
    
    return [main_q, followup_q]


# Template matching functions now imported from modules.template_catalog
# These functions handle field standardization against the template dictionary:
# - TemplateCatalog class for loading and matching templates
# - FindResult dataclass for match results
# - merge_with_template for merging parsed fields with templates
# - Helper functions for text normalization and matching

# ---------- Parsing

def _emit_parent_guardian_override(title: str, key: str, qtype: str, ctrl: Dict, section: str, insurance_scope: Optional[str], debug: bool):
    """Route parent/guardian-labeled fields to parent_* keys when appropriate."""
    if not PARENT_RE.search(title):
        return key, qtype, ctrl
    low = title.lower()
    if SSN_RE.search(low):
        if debug: print(f"  [debug] gate: parent_routing -> '{title}' -> parent_ssn")
        return "parent_ssn", "input", {"input_type":"ssn"}
    if PHONE_RE.search(low):
        if debug: print(f"  [debug] gate: parent_routing -> '{title}' -> parent_phone")
        return "parent_phone", "input", {"input_type":"phone"}
    if re.search(r"\b(name)\b", low):
        if debug: print(f"  [debug] gate: parent_routing -> '{title}' -> parent_name")
        return "parent_name", "input", {"input_type":"text"}
    return key, qtype, ctrl

def _insurance_scope_key(key: str, section: str, insurance_scope: Optional[str], title: str, debug: bool) -> str:
    """Attach __primary/__secondary to SSN/ID under insurance context; prefer patient-level elsewhere."""
    low = title.lower()
    if "insurance" in section.lower() or insurance_scope or re.search(r"\b(insured|subscriber|policy|member)\b", low):
        if key == "ssn" and insurance_scope:
            if debug: print(f"  [debug] gate: ssn_scoped -> '{title}' -> ssn{insurance_scope}")
            return f"ssn{insurance_scope}"
    return key

def _insurance_id_ssn_fanout(title: str) -> Optional[List[Tuple[str, str, Dict]]]:
    """Detect lines that contain both ID and SS tokens and fan-out to two controls."""
    t = _sanitize_words(title)
    has_id = bool(re.search(r"\b(id|member\s*id|policy\s*id|identification\s*number|subscriber\s*id|policy\s*#|member\s*#)\b", t))
    has_ss = bool(re.search(r"\b(ssn|social\s*security|ss#|soc\s*sec)\b", t))
    if has_id and has_ss:
        return [("insurance_id_number", "input", {"input_type":"text"}), ("ssn", "input", {"input_type":"ssn"})]
    return None


def detect_fill_in_blank_field(line: str, prev_line: Optional[str] = None, next_line: Optional[str] = None) -> Optional[Tuple[str, str]]:
    """
    Category 1 Fix 1.2: Detect standalone fill-in-blank fields.
    
    Pattern: Lines with mostly underscores (5+ consecutive underscores).
    Uses text on same line, previous line, or next line as label.
    
    Production Parity Fix: Also check next line for labels (common in signature blocks).
    
    Returns: (key, title) or None
    """
    # Check if line has 5+ consecutive underscores
    if not re.search(r'_{5,}', line):
        return None
    
    # IMPORTANT: If line has multiple underscore groups with text between them,
    # it's likely a multi-field line that should be handled by other splitters
    # Pattern: text___ text___ indicates multiple fields, not a single fill-in-blank
    underscore_groups = re.findall(r'_{5,}', line)
    if len(underscore_groups) > 1:
        # Check if there's meaningful text between the underscore groups
        # Split by underscores and check text chunks
        parts = re.split(r'_{5,}', line)
        text_chunks_between = [p.strip() for p in parts[1:-1] if p.strip()]  # Middle chunks
        # If there are text labels between underscore groups, this is multi-field
        if text_chunks_between and any(len(c) > 2 for c in text_chunks_between):
            return None  # Let multi-field splitters handle this
    
    # Count underscore vs non-underscore content
    underscore_count = line.count('_')
    non_underscore_chars = len([c for c in line if c not in ('_', ' ', '\t')])
    
    # If line is mostly underscores (2:1 ratio), it's a fill-in-blank
    if underscore_count < 5 or underscore_count < non_underscore_chars * 2:
        return None
    
    # Try to extract label from same line (before underscores)
    label_match = re.match(r'^([^_]+?)[:.]?\s*_{5,}', line)
    if label_match:
        label = label_match.group(1).strip()
        if label and len(label) >= 2:
            return (slugify(label), label)
    
    # PRODUCTION PARITY FIX: Try to use next line as label (common in signature blocks)
    # Pattern: blank underscore line followed by "Patient Signature" or similar
    if next_line:
        next_clean = next_line.strip().rstrip(':.')
        if next_clean and len(next_clean) < 100 and not re.search(CHECKBOX_ANY, next_clean):
            # Check if next line looks like a field label (not a sentence)
            # Common signature/name/date labels
            signature_patterns = [
                r'\b(?:patient|parent|guardian|witness|provider|doctor|dentist).*(?:signature|name|date)\b',
                r'\b(?:signature|name|date)\b',
                r'\bsign(?:ed)?\s+(?:by|on|at)\b'
            ]
            next_lower = next_clean.lower()
            is_label = any(re.search(p, next_lower) for p in signature_patterns)
            
            if is_label and not is_heading(next_clean):
                # Use next line as label
                return (slugify(next_clean), next_clean)
    
    # Try to use previous line as label if available
    if prev_line:
        prev_clean = prev_line.strip().rstrip(':.')
        if prev_clean and len(prev_clean) < 100 and not re.search(CHECKBOX_ANY, prev_clean):
            # Make sure it's not a heading or instructional text
            if not is_heading(prev_clean):
                # Archivev22 Enhancement: Don't use long descriptive sentences as labels for underscores
                # If the previous line is very long or looks like a sentence, skip it
                if len(prev_clean) < 80 and not (prev_clean.endswith('.') and len(prev_clean) > 50):
                    return (slugify(prev_clean), prev_clean)
    
    # Archivev22 Enhancement: Don't create generic placeholders for orphaned underscores
    # These are often signature lines or date lines without clear context
    # Only create them if the line has some identifying text
    line_text = re.sub(r'_{5,}', '', line).strip()
    if line_text and len(line_text) >= 2:
        # Has some text, create generic with that text
        return (slugify(line_text), line_text)
    
    # No label found and no identifying text - skip this underscore field
    return None


def detect_inline_checkbox_with_text(line: str) -> Optional[Tuple[str, str, str]]:
    """
    Detect inline checkboxes with continuation text.
    
    Example: "[ ] Yes, send me text alerts"
    Returns: ("yes_text_alerts", "Yes, send me text alerts", "radio")
    
    Priority 2.3: Inline Checkbox with Continuation Text
    - Detects checkboxes embedded mid-sentence
    - Extracts the boolean option and the continuation text
    - Pattern: "[ ] Yes/No, [descriptive text]"
    """
    # Archivev21 Fix 6: Make pattern more strict to avoid false positives
    # Require comma or "send" after Yes/No to ensure it's a continuation, not a separate field
    # Pattern: checkbox followed by Yes/No, then comma/space and continuation text
    pattern = r'^\s*' + CHECKBOX_ANY + r'\s*(Yes|No)[\s,]+(.*?)$'
    match = re.match(pattern, line, re.I)
    
    if not match:
        return None
    
    option = match.group(1).capitalize()  # "Yes" or "No"
    continuation = match.group(2).strip()
    
    # Need meaningful continuation text (at least 10 chars)
    # Archivev21 Fix 6: Also check that continuation doesn't look like a standalone field label
    # (i.e., doesn't start with a capital letter followed by lowercase and end with "Insurance", "Plan", etc.)
    if len(continuation) < 10:
        return None
    
    # Check if this looks like a false positive (e.g., "[ ] N o D ental Insurance")
    # If continuation starts with lowercase or has spaces between capital letters, likely OCR error
    if re.match(r'^[a-z]\s+[A-Z]', continuation):
        return None  # Likely "N o D ental" pattern - not a real continuation
    
    # Archivev21 Fix 6: Clean continuation text to fix any remaining OCR errors
    continuation_cleaned = clean_field_title(continuation)
    
    # Generate a descriptive key from the cleaned continuation
    key = slugify(continuation_cleaned)
    if len(key) > 40:
        key = key[:40]
    
    # Title includes both the option and the cleaned description
    title = f"{option}, {continuation_cleaned}"
    
    # Determine field type - usually a boolean/radio
    field_type = "radio"
    
    return (key, title, field_type)


def detect_multiple_label_colon_line(line: str) -> Optional[List[Tuple[str, str]]]:
    """
    Detect lines with multiple "Label:" patterns that should be split into separate fields.
    
    Examples:
        "Address: Apt# State: Zip:" -> [("address", "Address"), ("apt", "Apt#"), ("state", "State"), ("zip", "Zip")]
        "Name: DOB: SSN:" -> [("name", "Name"), ("dob", "DOB"), ("ssn", "SSN")]
        "Name of Insurance Company: Policy Holder Name: Member ID/SS#:" -> separate fields for each label with colon
    
    This handles cases where multiple form fields are on the same line, each with a colon.
    Also handles "Label: Field Label:" patterns where Field is between colons.
    """
    # Count colons in the line
    colon_count = line.count(':')
    
    # Need at least 2 colons to be a multi-label line
    if colon_count < 2:
        return None
    
    # Strategy: Split by colons and identify labels
    # Between colons, there might be multiple field labels
    # We identify these by looking for very short words or words with # that are clearly separate labels
    # But we keep multi-word phrases together (like "Policy Holder Name" or "Member ID")
    
    fields = []
    parts = line.split(':')
    
    for i in range(len(parts) - 1):  # -1 because last part after final colon has no field
        label = parts[i].strip()
        
        if i == 0:
            # First label - use it directly
            label = re.sub(r'[_/]+$', '', label).strip()
            if label and len(label) >= 2:
                field_key = slugify(label)
                fields.append((field_key, label))
        else:
            # Check if this part contains multiple CLEARLY SEPARATE field labels
            # Only split if we have obvious separate fields like "Apt# State" or "DOB SSN"
            
            # Split into tokens
            tokens = label.split()
            
            # Heuristic: Only split if:
            # 1. We have exactly 2 tokens AND both are short (<=5 chars) OR one has #
            # 2. We have multiple all-caps abbreviations
            should_split = False
            
            if len(tokens) == 2:
                # Two tokens - check if they look like separate fields
                has_special = any(re.search(r'#', t) for t in tokens)
                both_short = all(len(t) <= 5 for t in tokens)
                both_caps = all(t.isupper() and len(t) >= 2 for t in tokens)
                
                if (both_short and has_special) or both_caps:
                    should_split = True
            
            if should_split:
                # Split into separate fields
                for token in tokens:
                    token = token.strip()
                    if token and len(token) >= 2:
                        token = re.sub(r'[_/]+$', '', token).strip()
                        if token:
                            field_key = slugify(token)
                            fields.append((field_key, token))
            else:
                # Keep as single label (this preserves "Policy Holder Name", "Member ID/ SS#")
                label = re.sub(r'[_/]+$', '', label).strip()
                if label and len(label) >= 2:
                    field_key = slugify(label)
                    fields.append((field_key, label))
    
    # Return only if we found at least 2 valid fields
    if len(fields) >= 2:
        return fields
    
    return None


def detect_multi_field_line(line: str, section: str = "", prev_context: List[str] = None) -> Optional[List[Tuple[str, str]]]:
    """
    Improvement 3: Detect lines with a single label followed by multiple blank fields,
    with enhanced context awareness.
    
    Example: "Phone: Mobile _____ Home _____ Work _____"
    Returns: [("phone_mobile", "Mobile"), ("phone_home", "Home"), ("phone_work", "Work")]
    
    Improvement 3 Enhancement:
    - "Responsible Party: First ___ Last ___" in Insurance section
      → [("responsible_party_first_name", "First Name"), ("responsible_party_last_name", "Last Name")]
    - "Guardian Name: First ___ Last ___"
      → [("guardian_first_name", "First Name"), ("guardian_last_name", "Last Name")]
    
    Args:
        line: The line to parse
        section: Current section (e.g., "Insurance", "Patient Information")
        prev_context: List of previous lines for context extraction
    
    Priority 2.1: Multi-Field Label Splitting
    - Detects multiple underscores or blanks after a single label
    - Identifies common sub-field keywords (Home/Work/Cell, Mobile/Office, Primary/Secondary)
    - Uses spacing analysis to detect distinct blank columns
    """
    if prev_context is None:
        prev_context = []
    # Pattern: Label: followed by multiple keywords with blanks/underscores
    # Common keywords for sub-fields
    sub_field_keywords = {
        'mobile': 'mobile',
        'cell': 'mobile',
        'home': 'home',
        'work': 'work',
        'office': 'work',
        'primary': 'primary',
        'secondary': 'secondary',
        'personal': 'personal',
        'business': 'business',
        'other': 'other',
        'fax': 'fax',
        'preferred': 'preferred',
        # Patch 2: Add time-of-day keywords for phone fields
        'day': 'day',
        'evening': 'evening',
        'night': 'night',
        # Archivev21 Fix 3: Add common name field keywords
        'first': 'first',
        'last': 'last',
        'middle': 'middle',
        'mi': 'middle_initial',
        'nickname': 'nickname',
        'preferred': 'preferred',
    }
    
    # Look for a label (word/phrase ending with colon) followed by multiple sub-fields
    # Pattern: "Label: Keyword1 ____ Keyword2 ____ Keyword3 ____"
    label_match = re.match(r'^([A-Za-z\s]+?):\s+(.+)$', line)
    if not label_match:
        return None
    
    base_label = label_match.group(1).strip()
    remainder = label_match.group(2)
    
    # Improvement 3: Extract contextual prefix from label or previous lines
    context_prefix = None
    
    # Check if the label itself contains context keywords
    label_lower = base_label.lower()
    context_keywords = {
        'guardian': 'guardian',
        'responsible party': 'responsible_party',
        'emergency contact': 'emergency_contact',
        'insured': 'insured',
        'policy holder': 'policy_holder',
        'policyholder': 'policy_holder',
        'subscriber': 'subscriber',
        'patient': 'patient',
        'employer': 'employer',
        'spouse': 'spouse',
        'parent': 'parent',
        'child': 'child',
    }
    
    for keyword, prefix in context_keywords.items():
        if keyword in label_lower:
            context_prefix = prefix
            break
    
    # If no context in label, check previous 2 lines
    if not context_prefix and prev_context:
        for prev_line in prev_context[-2:]:
            prev_lower = prev_line.lower()
            for keyword, prefix in context_keywords.items():
                if keyword in prev_lower:
                    # Extract the context phrase before colon if present
                    if ':' in prev_line:
                        context_phrase = prev_line.split(':')[0].strip()
                        context_prefix = slugify(context_phrase)[:20]
                    else:
                        context_prefix = prefix
                    break
            if context_prefix:
                break
    
    # If in Insurance section and still no context, use "insurance" prefix for name fields
    if not context_prefix and section and 'insurance' in section.lower():
        if any(keyword in label_lower for keyword in ['name', 'insured', 'policy']):
            context_prefix = 'insurance'
    
    # Patch 2: Normalize slashes to spaces (e.g., "Day/Evening" -> "Day Evening")
    remainder = re.sub(r'/', ' ', remainder)
    
    # Look for multiple keywords separated by blanks/underscores
    # Pattern: keyword followed by blanks/underscores (at least 2)
    blank_pattern = r'[_\s]{2,}'
    
    # Split by significant blank sequences
    parts = re.split(blank_pattern, remainder)
    
    # Filter to get only meaningful keywords
    keywords_found = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Check if this part is a known sub-field keyword
        part_lower = part.lower()
        if part_lower in sub_field_keywords:
            keywords_found.append((part, sub_field_keywords[part_lower]))
    
    # Need at least 2 keywords to be a multi-field line
    if len(keywords_found) < 2:
        return None
    
    # Generate field descriptors
    base_key = slugify(base_label)
    result = []
    for display_name, suffix_key in keywords_found:
        # Improvement 3: Apply context prefix to create meaningful field keys
        if context_prefix:
            # If context is from the label itself (e.g., "Guardian Name"), 
            # and suffix is a name component, create contextual key
            if suffix_key in ['first', 'last', 'middle', 'middle_initial']:
                field_key = f"{context_prefix}_{suffix_key}_name"
            else:
                field_key = f"{context_prefix}_{suffix_key}"
        else:
            field_key = f"{base_key}_{suffix_key}"
        
        # Archivev21 Fix 3: For name fields, use simpler titles
        if base_label.lower() in ['patient name', 'name', 'insured name', "insured's name", 
                                   'guardian name', 'responsible party name', 'policy holder name']:
            # Use just the subfield name as title (e.g., "First Name", "Last Name")
            if suffix_key in ['first', 'last', 'middle', 'middle_initial']:
                field_title = f"{display_name.title()} Name" if display_name.lower() in ['first', 'last', 'middle'] else display_name.title()
            else:
                field_title = display_name.title()
        else:
            field_title = f"{base_label} ({display_name})"
        result.append((field_key, field_title))
    
    return result


def detect_inline_text_options(line: str) -> Optional[Tuple[str, str, List[Tuple[str, str]]]]:
    """
    Enhancement 2: Detect questions with inline text options.
    
    Patterns to detect:
    - "Question? Y or N" -> Yes/No options
    - "Question? Yes or No" -> Yes/No options
    - "Sex M or F" -> Male/Female options
    - "Gender: Male/Female" -> Male/Female options
    
    Returns:
        Tuple of (question_text, option_type, options) where options is [(name, value), ...]
        None if no inline options detected
    """
    line_stripped = line.strip()
    
    # Pattern 1: Y or N (with optional question mark and "If yes/no" continuation)
    yn_pattern = r'^(.+?)\s+([Yy]\s+or\s+[Nn])\s*(?:,?\s*[Ii]f\s+(?:yes|no).*)?$'
    yn_match = re.match(yn_pattern, line_stripped)
    if yn_match:
        question_text = yn_match.group(1)
        # Clean up the question text
        question_text = re.sub(r'\s+$', '', question_text)
        
        options = [
            ("Yes", "yes"),
            ("No", "no")
        ]
        return (question_text, "yes_no", options)
    
    # Pattern 2: Yes or No (full words)
    yesno_pattern = r'^(.+?)\s+(Yes\s+or\s+No)\s*(?:,?\s*[Ii]f\s+(?:yes|no).*)?$'
    yesno_match = re.match(yesno_pattern, line_stripped, re.I)
    if yesno_match:
        question_text = yesno_match.group(1)
        question_text = re.sub(r'\s+$', '', question_text)
        
        options = [
            ("Yes", "yes"),
            ("No", "no")
        ]
        return (question_text, "yes_no", options)
    
    # Pattern 3: M or F / Male or Female (Sex/Gender)
    sex_pattern = r'^(.+?)\s+(?:M\s+or\s+F|Male\s+or\s+Female|M/F|Male/Female)\s*'
    sex_match = re.match(sex_pattern, line_stripped, re.I)
    if sex_match:
        question_text = sex_match.group(1)
        # If question text is just "Sex" or "Gender", that's the label
        question_text = re.sub(r'[:\-]?\s*$', '', question_text)
        
        options = [
            ("Male", "male"),
            ("Female", "female"),
            ("Other", "other"),
            ("Prefer not to self identify", "not_say")
        ]
        return (question_text, "sex_gender", options)
    
    # Category 1 Fix 1.3: Pattern 4 - "Circle one:" or "Check one:" followed by space-separated options
    circle_pattern = r'^(.+?)(?:circle|check)\s+one:\s*(.+)$'
    circle_match = re.match(circle_pattern, line_stripped, re.I)
    if circle_match:
        question_text = circle_match.group(1).strip()
        options_text = circle_match.group(2).strip()
        
        # Split options by whitespace (2+ spaces) or common separators
        option_list = re.split(r'\s{2,}|,\s*|\|\s*', options_text)
        options = []
        for opt in option_list:
            opt = opt.strip()
            if opt and len(opt) > 1:  # Skip single chars
                # Create value from option text
                opt_value = opt.lower().replace(' ', '_').replace('-', '_')
                options.append((opt, opt_value))
        
        # Need at least 2 options
        if len(options) >= 2:
            # If question text is empty, use "Please select one"
            if not question_text:
                question_text = "Please select one"
            return (question_text, "select_one", options)
    
    # Category 1 Fix 1.3: Pattern 5 - Options separated by slashes without checkbox markers
    # e.g., "Marital Status: Single/Married/Divorced"
    slash_options_pattern = r'^(.+?):\s*([A-Za-z]+(?:/[A-Za-z]+){1,})\s*$'
    slash_match = re.match(slash_options_pattern, line_stripped)
    if slash_match:
        question_text = slash_match.group(1).strip()
        options_text = slash_match.group(2)
        
        # Split by slashes
        option_list = options_text.split('/')
        options = []
        for opt in option_list:
            opt = opt.strip()
            if opt:
                opt_value = opt.lower().replace(' ', '_')
                options.append((opt, opt_value))
        
        # Need at least 2 options and all options should be short (single words/phrases)
        if len(options) >= 2 and all(len(o[0]) < 20 for o in options):
            return (question_text, "select_one", options)
    
    return None


def detect_embedded_parenthetical_field(line: str) -> Optional[Tuple[str, str]]:
    """
    Detect fields embedded in sentences with parenthetical labels.
    
    Examples:
        "I, _____(print name) have been..." -> ("print_name", "Print Name")
        "PATIENT CONSENT: I, _____(print name) have..." -> ("patient_name", "Patient Name") 
        "Date: _____(mm/dd/yyyy)" -> ("date", "Date")
        "Signature: _____(patient signature)" -> ("signature", "Patient Signature")
    
    This pattern is common in consent forms where the field label is given
    as a clarification in parentheses after blank underscores.
    
    Production Parity Fix: Capture fields that were previously missed in consent forms.
    """
    # Pattern: underscores followed by parenthetical label
    # Need at least 3 underscores and a label in parentheses
    pattern = r'_{3,}\s*\(([^)]{3,40})\)'
    
    matches = list(re.finditer(pattern, line))
    if not matches:
        return None
    
    # Take the first match (most common case)
    match = matches[0]
    label_text = match.group(1).strip()
    
    # Clean up common variations
    label_lower = label_text.lower()
    
    # Map common variations to standard field names
    if 'print' in label_lower and 'name' in label_lower:
        # "print name", "please print name", "print full name"
        return ("patient_name", "Patient Name")
    elif 'name' in label_lower and 'patient' in label_lower:
        return ("patient_name", "Patient Name")
    elif 'date' in label_lower and ('mm' in label_lower or 'dd' in label_lower or 'yyyy' in label_lower):
        return ("date", "Date")
    elif 'signature' in label_lower:
        # Extract more specific signature type if present
        if 'patient' in label_lower:
            return ("patient_signature", "Patient Signature")
        elif 'witness' in label_lower:
            return ("witness_signature", "Witness Signature")
        elif 'guardian' in label_lower or 'parent' in label_lower:
            return ("guardian_signature", "Guardian Signature")
        else:
            return ("signature", "Signature")
    else:
        # Generic case: use the label text as-is
        # Clean and slugify the label
        clean_label = label_text.strip()
        # Remove common filler words
        clean_label = re.sub(r'\b(please|kindly)\b', '', clean_label, flags=re.I).strip()
        
        # Create title case for display
        title = ' '.join(word.capitalize() for word in clean_label.split())
        
        # Create key
        key = slugify(clean_label)
        
        # Only return if we have a reasonable key (not too short)
        if len(key) >= 3:
            return (key, title)
    
    return None


# ============================================================================
# SECTION 3: MAIN PARSING LOGIC
# ============================================================================
# Core text-to-JSON conversion logic. Orchestrates field detection and Question creation.

def parse_to_questions(text: str, debug: bool=False) -> List[Question]:
    # Improvement #2: Apply OCR corrections early in the pipeline
    text = clean_checkbox_ocr_artifacts(text)  # Clean checkbox artifacts first
    text = enhance_dental_term_corrections(text)
    text = correct_phone_number_patterns(text)
    text = correct_date_patterns(text)
    
    # Step 1: Clean headers/footers but DON'T normalize yet (preserves tabs for field splitting)
    lines = scrub_headers_footers(text)
    lines = coalesce_soft_wraps(lines)
    
    # Step 2: Split multi-field lines BEFORE normalizing (so tabs are preserved)
    pre_count = len(lines)
    lines = preprocess_lines(lines)
    post_count = len(lines)
    if post_count != pre_count and debug:
        print(f"  [debug] preprocess_lines: {pre_count} -> {post_count} lines")
    
    # Step 3: NOW normalize glyphs (after field splitting, so tabs don't get lost)
    lines = [normalize_glyphs_line(x) for x in lines]
    
    # Parity Fix: Detect if document is an information sheet (not a form)
    # Information sheets have instructional text but no fillable fields
    # Heuristics: 
    # 1. No common field markers (colons, underscores, blank lines for input)
    # 2. Short document (<20 lines) with mostly instructional content
    # 3. Contains congratulatory or informational keywords
    def is_information_sheet(text_lines: List[str]) -> bool:
        """Detect if document is informational content, not a form to fill out."""
        if len(text_lines) < 5:  # Too short to determine
            return False
        
        # Count field indicators
        field_indicators = 0
        info_indicators = 0
        
        for line in text_lines[:30]:  # Check first 30 lines
            line_lower = line.lower().strip()
            # Field indicators
            if re.search(r':\s*_{2,}', line):  # "Label: ___"
                field_indicators += 1
            if re.search(r'^\s*\[\s*\]', line):  # Checkbox at start
                field_indicators += 1
            if re.search(r'\b(name|address|phone|email|date|signature|birth|ssn):', line_lower):
                field_indicators += 1
                
            # Information indicators
            if re.match(r'^(congratulations|welcome|thank you|important|note|remember)', line_lower):
                info_indicators += 1
            if 'retainer' in line_lower and ('#' in line_lower or 'guide' in line_lower):
                info_indicators += 2  # Strong indicator of retainer instructions
            if re.search(r'\bhooray\b|beautiful smile|party.*over', line_lower):
                info_indicators += 1
            
        # Decision: If very few field indicators and some info indicators, it's informational
        if field_indicators <= 1 and info_indicators >= 2:
            return True
        if len(text_lines) < 20 and field_indicators == 0 and info_indicators >= 1:
            return True
        return False
    
    if is_information_sheet(lines):
        if debug:
            print("  [debug] Document detected as information sheet, not a fillable form. Returning empty field list.")
        # Return empty questions list - this is not a form to be filled
        return []

    questions: List[Question] = []
    cur_section = "General"
    seen_signature = False
    insurance_scope: Optional[str] = None  # "__primary" / "__secondary"

    i = 0
    while i < len(lines):
        raw = lines[i]
        line = collapse_spaced_caps(raw.strip())
        
        # Debug specific lines
        if debug and 'Appearance' in raw and 'Function' in raw:
            print(f"  [debug] PROCESSING line {i}: '{raw[:80]}' (section={cur_section})")
        
        if not line:
            i += 1; continue

        # NEW Improvement 1: Skip numbered list items (e.g., "(i)", "(ii)", "(vii)")
        # These are part of consent/terms text and should not be separate fields
        if is_numbered_list_item(line):
            if debug:
                print(f"  [debug] skipping numbered list item: '{line[:60]}'")
            i += 1
            continue

        # NEW Improvement 6: Skip form metadata (revision codes, copyright, etc.)
        # Form identifiers should not become fields
        if is_form_metadata(line):
            if debug:
                print(f"  [debug] skipping form metadata: '{line[:60]}'")
            i += 1
            continue

        # NEW Improvement 7: Skip practice location text (office addresses)
        # Practice addresses should not become data fields
        prev_context = lines[max(0, i-2):i] if i > 0 else []
        if is_practice_location_text(line, prev_context):
            if debug:
                print(f"  [debug] skipping practice location: '{line[:60]}'")
            i += 1
            continue

        # Improvement #7: Skip instructional paragraphs (consent/legal text)
        # Long paragraphs with legal/consent language should not become fields
        # Archivev23 Fix: Don't skip consent disclosure paragraphs - they should be captured as terms
        # Allow through if: in Consent section OR looks like consent/risk disclosure body text
        if is_instructional_paragraph(line):
            # Check if this might be consent body text that should be captured as terms
            # Consent body text typically: mentions risks, complications, procedures, treatments
            # AND is reasonably long (multiple sentences about medical/dental topics)
            consent_body_keywords = ['risk', 'complication', 'procedure', 'treatment', 'may include',
                                    'may result', 'may cause', 'may occur', 'include but not limited',
                                    'understand that', 'i consent', 'i acknowledge', 'informed about']
            has_consent_keywords = any(kw in line.lower() for kw in consent_body_keywords)
            has_medical_terms = any(term in line.lower() for term in ['endodontic', 'dental', 'tooth', 'surgery', 
                                                                      'anesthesia', 'medication', 'extraction'])
            is_consent_body = has_consent_keywords and (has_medical_terms or len(line) > 200)
            
            # If in Consent section or looks like consent body, allow it through for terms capture
            # Otherwise skip as regular instructional text
            if cur_section == "Consent" or is_consent_body:
                # Allow through - will be captured as terms field later
                pass
            else:
                if debug:
                    print(f"  [debug] skipping instructional text: '{line[:60]}...'")
                i += 1
                continue

        # Insurance anchoring
        if INSURANCE_BLOCK_RE.search(line):
            cur_section = "Insurance"; insurance_scope = None
        if INSURANCE_PRIMARY_RE.search(line):
            cur_section = "Insurance"; insurance_scope = "__primary"
        if INSURANCE_SECONDARY_RE.search(line):
            cur_section = "Insurance"; insurance_scope = "__secondary"

        # Archivev12 Fix: Check for special field patterns BEFORE heading detection
        # to prevent them from being treated as headings
        # Phase 4 Fix: Also check for preferred contact patterns
        is_special_sex_field = any(p.search(line) for p in SEX_GENDER_PATTERNS)
        is_special_marital_field = any(p.search(line) for p in MARITAL_STATUS_PATTERNS)
        is_special_preferred_contact = any(p.search(line) for p in PREFERRED_CONTACT_PATTERNS)
        is_special_field = is_special_sex_field or is_special_marital_field or is_special_preferred_contact
        
        # Fix 2: Section heading with multi-line header detection
        if not is_special_field and is_heading(line):
            # Look ahead to see if the next line is also a heading (multi-line header)
            # But only combine if the next line appears to be a continuation, not a new section
            potential_headers = [line]
            j = i + 1
            while j < len(lines) and j < i + 3:  # Look ahead up to 2 lines
                next_raw = lines[j]
                next_line = collapse_spaced_caps(next_raw.strip())
                if not next_line:
                    break
                # Archivev12 Fix: Don't include special fields in multi-line headers
                is_next_special_sex = any(p.search(next_line) for p in SEX_GENDER_PATTERNS)
                is_next_special_marital = any(p.search(next_line) for p in MARITAL_STATUS_PATTERNS)
                is_next_special = is_next_special_sex or is_next_special_marital
                
                # Fix: Only combine if next line appears to be a continuation
                # Don't combine if next line has "information", "practice", "consent" which are typically new sections
                # Don't combine if next line contains field-like patterns (colons with short text)
                # Only combine if starts with lowercase (continuation) or is very short descriptive phrase
                has_section_keywords = re.search(r'\b(information|practice|consent|authorization|attestation|form|release)\b', next_line, re.I)
                has_field_pattern = next_line.endswith(':') and len(next_line.split()) <= 4
                
                is_continuation = (
                    is_heading(next_line) and 
                    not is_next_special and
                    not has_section_keywords and
                    not has_field_pattern and
                    next_line[0].islower()  # Only combine if starts lowercase (true continuation)
                )
                
                if is_continuation:
                    potential_headers.append(next_line)
                    j += 1
                else:
                    break
            
            # If we found multiple consecutive headings, combine them
            if len(potential_headers) > 1:
                combined = " ".join(potential_headers)
                new_section = normalize_section_name(combined)
                cur_section = new_section
                if debug:
                    print(f"  [debug] multi-line header: {potential_headers} -> {cur_section}")
                i = j  # Skip past all the header lines
                continue
            else:
                new_section = normalize_section_name(line)
                # Archivev10 Fix: Don't override Medical/Dental History sections with "General"
                # This prevents random headings within those sections from changing the section
                # But DO allow changing between specific sections
                if cur_section in {"Medical History", "Dental History"} and new_section == "General":
                    # Keep the current specific section, don't change to General
                    pass
                else:
                    cur_section = new_section
            
            low = line.lower()
            if "insurance" in low:
                if "primary" in low: insurance_scope = "__primary"
                elif "secondary" in low: insurance_scope = "__secondary"
                else: insurance_scope = None
            else:
                insurance_scope = None
            i += 1; continue
        
        # Archivev10 Fix 2: Enhanced category header detection
        # Check if this is a category header that precedes a multi-column grid
        if i + 1 < len(lines):
            next_line = lines[i + 1]
            is_cat_header = is_category_header(line, next_line)
            if debug and is_cat_header:
                print(f"  [debug] is_category_header=True for line {i}: '{line[:60]}' (section={cur_section})")
            if is_cat_header:
                # Before skipping, check if this is actually a multi-column grid header
                # (e.g., "Appearance    Function    Habits")
                if cur_section in {"Medical History", "Dental History"}:
                    # Check if line has multiple column-like parts and next line has multiple checkboxes
                    parts = re.split(r'\s{5,}', line.strip())
                    next_checkboxes = len(list(re.finditer(CHECKBOX_ANY, next_line)))
                    
                    if debug:
                        print(f"  [debug] category header check: '{line[:60]}' - parts={len(parts)}, next_cb={next_checkboxes}")
                    
                    if len(parts) >= 3 and next_checkboxes >= 3:
                        # This is a multi-column grid header! Try to detect and parse the grid
                        if debug:
                            print(f"  [debug] attempting multicolumn_grid detection from category header line {i}")
                        multicolumn_grid = detect_multicolumn_checkbox_grid(lines, i, cur_section)
                        if multicolumn_grid:
                            grid_question = parse_multicolumn_checkbox_grid(lines, multicolumn_grid, debug)
                            if grid_question:
                                questions.append(grid_question)
                                i = max(multicolumn_grid['data_lines']) + 1
                                continue
                        elif debug:
                            print(f"  [debug] multicolumn_grid detection failed")
                
                # If not a grid header, skip as before
                if debug:
                    print(f"  [debug] skipping category header: '{line[:60]}'")
                i += 1
                continue

        # Improvement 4: Check for medical conditions grid pattern
        # This detects "Do you have any of the following?" followed by many checkbox lines
        medical_grid = detect_medical_conditions_grid(lines, i, debug)
        if medical_grid:
            # Create a multi-select dropdown question with all the conditions
            key = slugify(medical_grid['title'])
            if len(key) > 50:
                key = "medical_conditions" if "medical" in cur_section.lower() else "dental_conditions"
            
            questions.append(Question(
                key,
                medical_grid['title'],
                cur_section if cur_section else "Medical History",
                "dropdown",
                control={"options": medical_grid['options'], "multi": True}
            ))
            
            if debug:
                print(f"  [debug] medical_grid -> '{medical_grid['title']}' with {len(medical_grid['options'])} options")
            
            # Skip past the grid
            i = medical_grid['end_idx'] + 1
            continue

        # Archivev8 Fix 1: Check for orphaned checkbox pattern
        # Use raw line (not collapsed) to preserve spacing
        if has_orphaned_checkboxes(raw) and i + 1 < len(lines):
            next_line = lines[i + 1]
            orphaned_options = associate_orphaned_labels_with_checkboxes(raw, next_line)
            
            if orphaned_options and len(orphaned_options) >= 2:
                # Found orphaned labels! Create a medical conditions question
                # Look back for a title
                title = None
                if i > 0 and len(lines[i-1].strip()) < 100:
                    prev_stripped = collapse_spaced_caps(lines[i-1].strip())
                    if prev_stripped and not re.search(CHECKBOX_ANY, prev_stripped):
                        title = prev_stripped.rstrip(':?.')
                
                if not title:
                    title = "Medical Conditions"
                
                key = slugify(title)
                q = Question(
                    key,
                    title,
                    cur_section,
                    "dropdown",
                    control={"options": [make_option(name, checked) for name, checked in orphaned_options], "multi": True}
                )
                questions.append(q)
                
                if debug:
                    print(f"  [debug] gate: orphaned_labels -> '{title}' with {len(orphaned_options)} options")
                
                # Skip the next line since we consumed it
                i += 2
                continue

        # --- NEW: Medical History Multi-select block ---
        if cur_section in {"Medical History", "Dental History"}:
            if re.search(r"\b(have you ever had|do you have|are you taking)\b", line, re.I) and not extract_compound_yn_prompts(line):
                main_prompt_title = line
                options: List[Tuple[str, Optional[bool]]] = []
                k = i + 1
                while k < len(lines) and lines[k].strip() and not is_heading(lines[k]):
                    option_line = lines[k]
                    
                    # Fix 3: Skip category headers within medical blocks
                    if k + 1 < len(lines) and is_category_header(option_line, lines[k + 1]):
                        k += 1
                        continue
                    
                    # Archivev8 Fix 1: Check for orphaned checkboxes within the medical history block
                    if has_orphaned_checkboxes(lines[k]) and k + 1 < len(lines):
                        orphaned_opts = associate_orphaned_labels_with_checkboxes(lines[k], lines[k + 1])
                        if orphaned_opts:
                            options.extend(orphaned_opts)
                            k += 2  # Skip both the checkbox line and label line
                            continue
                    
                    # Try compound Y/N prompts first (e.g., "Question? [ ] Yes [ ] No")
                    prompts = extract_compound_yn_prompts(option_line)
                    if prompts:
                        for p in prompts:
                            options.append((p, None))
                    else:
                        # Try inline options (e.g., "[ ] Option1 [ ] Option2")
                        inline_opts = options_from_inline_line(option_line)
                        if inline_opts:
                            options.extend(inline_opts)
                    
                    k += 1

                if len(options) >= 3: # Threshold to be considered a conditions block
                    control = {"options": [make_option(n, b) for n,b in options], "multi": True}
                    key = "medical_conditions" if cur_section == "Medical History" else "dental_conditions"
                    questions.append(Question(key, main_prompt_title, cur_section, "dropdown", control=control))
                    i = k
                    continue

        # Drop witness (unless it's a tabulated signature line with witness field)
        if WITNESS_RE.search(line) and '\t' not in line:
            i += 1; continue
        
        # Check for tabulated witness signature lines
        if WITNESS_RE.search(line) and '\t' in line:
            witness_fields = parse_tabulated_signature_line(line)
            if witness_fields:
                for field_dict in witness_fields:
                    q = Question(
                        field_dict['key'],
                        field_dict['title'],
                        field_dict.get('section', 'Consent'),
                        field_dict['type'],
                        control=field_dict.get('control', {})
                    )
                    questions.append(q)
                i += 1; continue

        # Signature (+ optional date)
        if "signature" in line.lower() or "signatory" in line.lower():
            # PARITY FIX: Skip if this is just a reference to signature in a paragraph (not an actual signature field)
            # Signature fields are typically short and contain underscores or are at the end of forms
            # Skip if the line is long (>100 chars) and doesn't have underscores (likely prose)
            line_stripped = line.strip()
            if len(line_stripped) > 100 and '_' not in line_stripped and not line.strip().endswith(':'):
                # This is likely a paragraph mentioning signature, not a signature field
                i += 1; continue
            
            # PARITY FIX: Skip lines like "Or authorized signatory" (orphaned text)
            if line_stripped.lower().startswith('or ') and len(line_stripped) < 40:
                i += 1; continue
            
            # PARITY FIX: Check if this is a tab-separated signature line with name/date fields
            # This check should happen for ALL signature lines, not just the first one
            tabulated_fields = parse_tabulated_signature_line(line)
            if tabulated_fields:
                # Add all the parsed fields (name, signature, date)
                for field_dict in tabulated_fields:
                    q = Question(
                        field_dict['key'],
                        field_dict['title'],
                        field_dict.get('section', 'Consent'),
                        field_dict['type'],
                        control=field_dict.get('control', {})
                    )
                    questions.append(q)
                seen_signature = True
                i += 1; continue
            
            # Fall back to single-field signature parsing
            if not seen_signature:
                # Original single-field signature logic
                questions.append(Question("signature", line.rstrip(":"), "Signature", "signature"))
                seen_signature = True
                # Adjacent Date (normalized title)
                if i + 1 < len(lines) and DATE_LABEL_RE.search(lines[i+1]):
                    title_dt = "Date Signed"
                    questions.append(Question(slugify(title_dt), title_dt, "Signature", "date", control={"input_type":"past"}))
                    i += 1
            i += 1; continue

        # Archivev10 Fix 1: Try multi-column checkbox grid detection first
        # This handles grids with 3+ checkboxes per line (common in medical/dental forms)
        if cur_section in {"Medical History", "Dental History"}:
            # Check if this line or upcoming lines form a multi-column checkbox grid
            multicolumn_grid = detect_multicolumn_checkbox_grid(lines, i, cur_section)
            if multicolumn_grid:
                grid_question = parse_multicolumn_checkbox_grid(lines, multicolumn_grid, debug)
                if grid_question:
                    questions.append(grid_question)
                    # Skip past the grid
                    i = max(multicolumn_grid['data_lines']) + 1
                    continue
            
            # Also check if this is a category header line followed by a grid
            # (e.g., "Appearance    Function    Habits    Previous Comfort Options")
            if not re.search(CHECKBOX_ANY, line):
                # Check if it looks like multiple column headers
                parts = re.split(r'\s{5,}', line.strip())
                if len(parts) >= 3 and all(len(p.split()) <= 4 for p in parts):
                    # Might be category headers, check if next line has multiple checkboxes
                    if i + 1 < len(lines):
                        next_checkboxes = len(list(re.finditer(CHECKBOX_ANY, lines[i + 1])))
                        if next_checkboxes >= 3:
                            # This is a category header line before a grid!
                            multicolumn_grid = detect_multicolumn_checkbox_grid(lines, i, cur_section)
                            if multicolumn_grid:
                                grid_question = parse_multicolumn_checkbox_grid(lines, multicolumn_grid, debug)
                                if grid_question:
                                    questions.append(grid_question)
                                    i = max(multicolumn_grid['data_lines']) + 1
                                    continue
        
        # Fix 4: Enhanced Table/Grid Detection - try multi-row table first
        table_info = detect_table_layout(lines, i)
        if table_info:
            # Parse entire table at once
            table_questions = parse_table_to_questions(lines, table_info, cur_section)
            questions.extend(table_questions)
            
            # Skip past the table
            i = max(table_info['data_lines']) + 1
            continue

        # Grid (existing logic for simpler cases)
        hdr_cols = looks_like_grid_header(line)
        if hdr_cols:
            ncols = len(hdr_cols)
            col_options: List[List[Tuple[str, Optional[bool]]]] = [[] for _ in range(ncols)]
            k = i + 1
            while k < len(lines) and lines[k].strip():
                row = collapse_spaced_caps(lines[k])
                if is_heading(row): break
                cells = chunk_by_columns(row, ncols)
                for cidx, cell in enumerate(cells):
                    if not cell.strip(): continue
                    opts=[]
                    if BULLET_RE.match(cell): 
                        o = option_from_bullet_line(cell); 
                        if o: opts.append(o)
                    else:
                        opts += options_from_inline_line(cell)
                    if not opts and ("," in cell or "/" in cell or ";" in cell):
                        for tok in re.split(r"[,/;]", cell):
                            tok = clean_token(tok)
                            if tok: opts.append((tok, None))
                    col_options[cidx].extend(opts)
                k += 1
            for cidx, col in enumerate(hdr_cols):
                opts = col_options[cidx]
                if not opts: continue
                options = [make_option(n,b) for (n,b) in opts]
                control = {"options": options, "multi": True}
                qkey = slugify(col)
                if insurance_scope and "insurance" in cur_section.lower():
                    qkey = f"{qkey}{insurance_scope}"
                questions.append(Question(qkey, col, cur_section, "dropdown", control=control))
            i = k; continue

        # --- Insurance ID/SSN fan-out (before other logic) ---
        fan = _insurance_id_ssn_fanout(line)
        if fan:
            for (kname, ktype, kctrl) in fan:
                qkey = _insurance_scope_key(kname, cur_section, insurance_scope, line, debug)
                questions.append(Question(qkey, CANON_LABELS.get(kname, kname.replace("_"," ").title()), "Insurance" if "insurance" in cur_section else cur_section, ktype, control=kctrl))
            if debug: print(f"  [debug] gate: insurance_fanout -> '{line}' -> ['insurance_id_number','ssn(+scope)']")
            i += 1; continue

        # Archivev15 Fix 1: Check for field label with inline checkbox option
        # Pattern: "Field Label:    <spaces>    [ ] Option text"
        # This must be checked BEFORE try_split_known_labels which incorrectly splits these lines
        # Use raw line (not collapsed) to preserve spacing
        field_checkbox_split = detect_field_with_inline_checkbox(raw)
        if field_checkbox_split:
            field_label, checkbox_option = field_checkbox_split
            
            # Create the main field (e.g., "Cell Phone")
            field_title = field_label.strip()
            field_key = slugify(field_title)
            field_itype = classify_input_type(field_title)
            
            questions.append(Question(
                field_key,
                field_title,
                cur_section,
                "input",
                control={"input_type": field_itype or "text"}
            ))
            
            # Create the checkbox field (e.g., "Yes, send me Text Message alerts")
            checkbox_title = checkbox_option.strip()
            # Use a prefix to ensure unique key that won't conflict with main field
            checkbox_key = f"opt_in_{slugify(checkbox_title)}"
            
            questions.append(Question(
                checkbox_key,
                checkbox_title,
                cur_section,
                "radio",
                control={"options": [make_option("Yes", True), make_option("No", False)]}
            ))
            
            if debug:
                print(f"  [debug] gate: field_with_inline_checkbox -> '{field_title}' + '{checkbox_title}'")
            
            i += 1
            continue

        # --- Composite / multi-label fan-out ---
        composite_labels = try_split_known_labels(line)
        if composite_labels:
            for ttl in composite_labels:
                if ttl.lower().startswith("state"):
                    qtype, ctrl = "states", {}
                elif "date" in ttl.lower() or "birth" in ttl.lower() or "dob" in ttl.lower():
                    qtype, ctrl = "date", {"input_type": classify_date_input(ttl)}
                else:
                    itype = classify_input_type(ttl)
                    qtype, ctrl = "input", ({"input_type": itype} if itype else {})
                key = slugify(ttl)
                # parent/guardian override
                key, qtype, ctrl = _emit_parent_guardian_override(ttl, key, qtype, ctrl, cur_section, insurance_scope, debug)
                # insurance scoping for SSN
                key = _insurance_scope_key(key, cur_section, insurance_scope, ttl, debug)
                if insurance_scope and "insurance" in cur_section.lower() and not key.endswith((PRIMARY_SUFFIX, SECONDARY_SUFFIX)) and key in {"ssn","insurance_id_number"}:
                    key = f"{key}{insurance_scope}"
                questions.append(Question(key, ttl, cur_section, qtype, control=ctrl))
            i += 1; continue

        # Performance Recommendation #3: Enhanced inline checkbox detection
        checkbox_result = detect_inline_checkbox_options(line)
        if checkbox_result:
            field_label, options = checkbox_result
            title = clean_field_title(field_label)
            key = slugify(title)
            # Determine if radio or checkbox based on context
            from .modules.performance_enhancements import infer_radio_vs_checkbox
            field_type = infer_radio_vs_checkbox(options, title)
            control = {"options": options}
            if field_type == "checkbox":
                control["multi"] = True
            questions.append(Question(key, title, cur_section, field_type, control=control))
            if debug: print(f"  [debug] enhanced_checkbox -> '{title}' with {len(options)} options ({field_type})")
            i += 1; continue

        # Single-checkbox → boolean radio (e.g., Responsible Party …)
        mbox = SINGLE_BOX_RE.match(line)
        if mbox and RESP_PARTY_RE.search(line):
            title = "Responsible party is someone other than patient?"
            key = slugify(title)
            if insurance_scope and "insurance" in cur_section.lower():
                key = f"{key}{insurance_scope}"
            control = {"options":[make_option("Yes",True), make_option("No",False)]}
            questions.append(Question(key, title, "Insurance", "radio", control=control))
            if debug: print(f"  [debug] gate: bool_single_box -> '{line}' -> radio Yes/No")
            i += 1; continue

        # Fix 4: Check for orphaned checkboxes (enhanced)
        if i + 1 < len(lines):
            orphaned_opts = extract_orphaned_checkboxes_and_labels(line, lines[i+1])
            if orphaned_opts:
                # Check if in medical/dental history section for condition collection
                is_condition_block = cur_section in {"Medical History", "Dental History"}
                
                # Try to find a better title from previous line
                title = "Please select all that apply:"
                if i > 0:
                    prev_line = collapse_spaced_caps(lines[i-1].strip())
                    if prev_line and not re.search(CHECKBOX_ANY, prev_line) and not is_heading(prev_line):
                        # Use previous line as title if it looks like a question
                        if prev_line.endswith('?') or prev_line.endswith(':') or len(prev_line) > 20:
                            title = prev_line.rstrip(':').strip()
                
                if is_condition_block:
                    # Add to a medical conditions dropdown
                    control = {"options": [make_option(n, b) for n, b in orphaned_opts], "multi": True}
                    key = "medical_conditions" if cur_section == "Medical History" else "dental_conditions"
                    questions.append(Question(key, title, cur_section, "dropdown", control=control))
                else:
                    # Create standalone dropdown
                    key = slugify(title)
                    control = {"options": [make_option(n, b) for n, b in orphaned_opts], "multi": True}
                    questions.append(Question(key, title, cur_section, "dropdown", control=control))
                
                if debug: print(f"  [debug] gate: orphaned_checkboxes -> found {len(orphaned_opts)} options")
                i += 2  # Skip both current and next line
                continue

        # Bullet -> terms (explanatory lists)
        if line.lstrip().startswith(("•","·")):
            terms_lines = [line]; k = i+1
            while k < len(lines) and lines[k].strip() and not is_heading(lines[k]):
                if lines[k].lstrip().startswith(("•","·")):
                    terms_lines.append(lines[k])
                else:
                    terms_lines[-1] += " " + lines[k].strip()
                k += 1
            txt_terms = " ".join(collapse_spaced_caps(x).strip() for x in terms_lines)
            questions.append(Question(
                slugify("terms_"+(terms_lines[0][:20] if terms_lines else "list")),
                "Terms", "Consent", "terms",
                control={"agree_text":"I have read and agree to the terms.","html_text":txt_terms}
            ))
            i = k; continue

        # Special: “Please share the following dates”
        if re.search(r"please\s+share\s+the\s+following\s+dates", line, re.I):
            harvested = " ".join(collapse_spaced_caps(lines[j].strip()) for j in range(i, min(i+3,len(lines))))
            for lbl in ["Cleaning","Cancer Screening","X-Rays","X Rays","Xray"]:
                if re.search(lbl.replace(" ", r"\s*"), harvested, re.I):
                    questions.append(Question(slugify(lbl), lbl, cur_section, "date", control={"input_type":"past"}))
            i += 1; continue

        # Robust 1..10 detection (pain scale etc.)
        digits = [int(x) for x in re.findall(r"\b\d+\b", line)]
        if digits and digits == list(range(1, 11)):
            title = line
            options = [{"name": str(n), "value": n} for n in range(1,11)]
            questions.append(Question(slugify(title), title, cur_section, "radio", control={"options": options}))
            i += 1; continue

        # Comms consent
        if re.search(r"\bsend me\b.*\btext\b", line, re.I):
            questions.append(Question("consent_text_alerts","Consent to Text Message alerts",cur_section,"radio",
                                      control={"options":[make_option("Yes",True),make_option("No",False)]}))
            i += 1; continue
        if re.search(r"\bsend me\b.*\bemail\b", line, re.I):
            questions.append(Question("consent_email_alerts","Consent to Email alerts",cur_section,"radio",
                                      control={"options":[make_option("Yes",True),make_option("No",False)]}))
            i += 1; continue

        # Fix 2: Enhanced "If Yes" Detection - try new pattern first
        question_text, has_followup = extract_yn_with_followup(line)
        if question_text and has_followup:
            # Create both question and follow-up using the new helper
            new_questions = create_yn_question_with_followup(question_text, cur_section)
            questions.extend(new_questions)
            if debug: print(f"  [debug] gate: yn_with_followup -> '{question_text}' + explanation field")
            i += 1
            continue
        
        # Compound Yes/No on one line (existing logic)
        compound_prompts = extract_compound_yn_prompts(line)
        emitted_compound = False
        if compound_prompts:
            for ptxt in compound_prompts:
                # Fix: Strip follow-up instructions from prompt to get clean question
                # Pattern: "Question? If yes, please explain:______" -> "Question?"
                clean_ptxt = re.sub(r'\s+(if\s+yes|if\s+so|please\s+explain|explain\s+below).*$', '', ptxt, flags=re.I).strip()
                if not clean_ptxt:
                    clean_ptxt = ptxt  # Fallback if regex removes everything
                
                key = slugify(clean_ptxt)
                if insurance_scope and "insurance" in cur_section.lower():
                    key = f"{key}{insurance_scope}"
                control = {"options":[make_option("Yes",True),make_option("No",False)]}
                
                # Fix 2: Enhanced follow-up field detection
                create_follow_up = False
                if re.search(IF_GUIDANCE_RE, line):
                    control["extra"] = {"type":"Input","value":True,"optional":True,"hint":"If yes, please explain"}
                    create_follow_up = True
                
                # Check same line for "if yes, please explain"
                if re.search(r'\b(if\s+yes|please\s+explain|if\s+so|explain\s+below)\b', line, re.I):
                    create_follow_up = True
                
                # Check next line for follow-up indicators
                if i + 1 < len(lines):
                    next_line = collapse_spaced_caps(lines[i+1].strip())
                    if re.search(r'^\s*(if\s+yes|please\s+explain|explain|comment|list|details?)', next_line, re.I):
                        create_follow_up = True
                
                # Create follow-up field if needed with conditional
                if create_follow_up:
                    follow_up_key = f"{key}_explanation"
                    # Check if not already created
                    if not any(q.key == follow_up_key for q in questions):
                        follow_up_q = Question(
                            follow_up_key,
                            "Please explain",
                            cur_section,
                            "input",
                            control={"input_type": "text", "hint": "Please provide details"}
                        )
                        # Add conditional to only show if Yes
                        follow_up_q.conditional_on = [(key, "yes")]
                        questions.append(follow_up_q)
                
                questions.append(Question(key, clean_ptxt, cur_section, "radio", control=control))
                emitted_compound = True
            if re.search(r"name\s+of\s+school", line, re.I):
                questions.append(Question("name_of_school","Name of School",cur_section,"input",control={"input_type":"text"}))
        if emitted_compound:
            i += 1; continue

        # Enhancement 2: Enhanced inline option detection
        # Detect questions with inline text options: "Question? Y or N", "Sex M or F", etc.
        inline_option_match = detect_inline_text_options(line)
        if inline_option_match:
            question_text, option_type, options = inline_option_match
            
            # Create question from the text before the options
            title = question_text.strip()
            if title.endswith('?'):
                title = title[:-1].strip()
            
            key = slugify(title)
            
            # Create control with detected options
            control = {"options": [{"name": opt[0], "value": opt[1]} for opt in options]}
            
            questions.append(Question(key, title, cur_section, "radio", control=control))
            
            if debug:
                print(f"  [debug] gate: inline_text_options -> '{title}' with {len(options)} options")
            
            i += 1
            continue
        
        # Archivev12 Fix 3: Special handling for Sex/Gender with text options (M or F) - LEGACY fallback
        # Keep for backward compatibility, but the new detect_inline_text_options should catch these
        sex_match = re.search(r'\b(sex|gender)\s*[:\-]?\s*(?:M\s*or\s*F|M/F|Male/Female|Mor\s*F)', line, re.I)
        if sex_match:
            key = "sex"
            title = sex_match.group(1).title()
            control = {
                "options": [
                    {"name": "Male", "value": "male"},
                    {"name": "Female", "value": "female"},
                    {"name": "Other", "value": "other"},
                    {"name": "Prefer not to self identify", "value": "not_say"}
                ]
            }
            questions.append(Question(key, title, cur_section, "radio", control=control))
            i += 1
            continue
        
        # Archivev12 Fix 4: Special handling for "Please Circle One:" marital status
        marital_match = re.search(r'(?:please\s+)?circle\s+one\s*:?\s*(.*?)$', line, re.I)
        if marital_match:
            # Extract options from the rest of the line
            options_text = marital_match.group(1).strip()
            # Common marital status options
            marital_options = []
            for opt in ['Single', 'Married', 'Divorced', 'Separated', 'Widowed', 'Widow']:
                if opt.lower() in options_text.lower():
                    if opt == 'Widow':
                        opt = 'Widowed'  # Normalize
                    if opt not in [o['name'] for o in marital_options]:
                        marital_options.append({"name": opt, "value": opt.lower()})
            
            if marital_options:
                # Add standard options if not present
                if not any(o['name'] == 'Prefer not to say' for o in marital_options):
                    marital_options.append({"name": "Prefer not to say", "value": "not say"})
                
                control = {"options": marital_options}
                questions.append(Question("marital_status", "Marital Status", cur_section, "radio", control=control))
                i += 1
                continue


        # Option harvesting for a single prompt (incl. “hear about us”)
        opts_inline = options_from_inline_line(line)
        opts_block: List[Tuple[str, Optional[bool]]] = []
        j = i + 1
        # collect bullets immediately below
        while j < len(lines):
            cand = collapse_spaced_caps(lines[j])
            if not cand.strip(): break
            o = option_from_bullet_line(cand)
            if o: opts_block.append(o); j += 1; continue
            # NEW: Also collect inline checkbox options from following lines (Fix 1 enhancement)
            inline_opts = options_from_inline_line(cand)
            if inline_opts:  # Any checkboxes found
                # Archivev20 Fix 5: Don't collect options from a line that has its own label
                # (e.g., "Marital Status: [ ] Married [ ] Single" should not be collected by "Ext#")
                bracket_pos = cand.find('[')
                
                # Archivev20 Fix 10: Enhanced label detection - check for label even without colon
                # Patterns: "Sex [ ] Male [ ] Female" or "Marital Status: [ ] Married"
                line_has_own_label = False
                if bracket_pos > 0:
                    text_before_bracket = cand[:bracket_pos].strip()
                    # Has label if: contains colon OR has meaningful text (3+ chars, not just numbers/symbols)
                    if ':' in text_before_bracket:
                        line_has_own_label = True
                    elif len(text_before_bracket) >= 3 and re.search(r'[A-Za-z]{3,}', text_before_bracket):
                        # Has at least 3 letters in a row (meaningful word), likely a label
                        line_has_own_label = True
                
                # Archivev20 Fix 8: Also don't collect if line starts with checkbox (standalone checkbox field)
                # (e.g., "[ ] Yes, send me Text Message alerts" should not be collected by "Zip:")
                line_starts_with_checkbox = bracket_pos <= 2  # Allow for leading whitespace
                
                if line_has_own_label or (len(inline_opts) == 1 and line_starts_with_checkbox):
                    # This line has its own label OR is a standalone checkbox field, don't collect
                    break
                
                if len(inline_opts) >= 2:  # Multiple options on this line - definitely collect
                    opts_block.extend(inline_opts)
                    j += 1
                    continue
                elif len(inline_opts) == 1 and opts_block:  # Single option but we already have options - continue collecting
                    opts_block.extend(inline_opts)
                    j += 1
                    continue
            # No valid options found on this line - check if it's a continuation or break
            if not re.search(CHECKBOX_ANY, cand):  # No checkboxes at all
                break
            # Has checkboxes but no valid labels - might be orphaned checkboxes, continue
            j += 1
            continue

        title = line.rstrip(":").strip()
        is_hear = bool(HEAR_ABOUT_RE.search(title))
        
        # Fix 1: If current line starts with checkbox and we have opts_block, look back for title
        if opts_block and re.match(r'^\s*' + CHECKBOX_ANY, line):
            if i > 0:
                prev_line = collapse_spaced_caps(lines[i-1].strip())
                if prev_line and not re.search(CHECKBOX_ANY, prev_line) and not is_heading(prev_line):
                    title = prev_line.rstrip(':').strip()
                    is_hear = bool(HEAR_ABOUT_RE.search(title))

        # “How did you hear...” — aggressively collect same line + next two non-heading lines
        if is_hear and not (opts_inline or opts_block):
            # Fix 1: Check if next line has inline checkboxes - if so, skip and let next line handle it
            # Fix 1: Check if next line(s) have inline checkboxes - skip blanks
            check_idx = i + 1
            while check_idx < len(lines) and not lines[check_idx].strip():
                check_idx += 1  # Skip blank lines
            if check_idx < len(lines):
                next_line_check = lines[check_idx].strip()
                if next_line_check and re.search(CHECKBOX_ANY, next_line_check):
                    next_opts_check = options_from_inline_line(next_line_check)
                    if len(next_opts_check) >= 2:
                        # Next line will handle this, skip for now
                        i += 1
                        continue
            
            # same line split
            for tok in re.split(r"[,/;]", title):
                tok = clean_token(tok)
                if tok and tok.lower() not in {"how did you hear about us", "referred by"}:
                    opts_inline.append((tok, None))
            k = i + 1
            extra_lines = 0
            while k < len(lines) and extra_lines < 2:
                cand = collapse_spaced_caps(lines[k]).strip()
                if not cand or is_heading(cand): break
                ob = option_from_bullet_line(cand)
                if ob:
                    opts_block.append(ob); k += 1; continue
                for tok in re.split(r"[,/;]", cand):
                    tok = clean_token(tok)
                    if tok: opts_block.append((tok, None))
                k += 1; extra_lines += 1
            j = max(j, k)

        # Check for multiple "Label:" patterns on the same line first
        # This handles cases like "Address: Apt# State: Zip:" or "Name: DOB: SSN:"
        multiple_label_fields = detect_multiple_label_colon_line(line)
        if multiple_label_fields:
            if debug:
                print(f"  [debug] multiple-label-colon detected: {line[:60]}... -> {len(multiple_label_fields)} fields")
                print(f"  [debug]   Keys: {[k for k, _ in multiple_label_fields]}")
            for field_key, field_title in multiple_label_fields:
                # Determine input type based on field name
                input_type = "text"
                if "phone" in field_key.lower():
                    input_type = "phone"
                elif "email" in field_key.lower():
                    input_type = "email"
                elif "zip" in field_key.lower() or field_key.lower() == "zip":
                    input_type = "zip"
                elif "ssn" in field_key.lower() or field_key.lower() in ["ss", "social_security"]:
                    input_type = "ssn"
                elif field_key.lower() in ["state", "st"]:
                    # Create a states field instead
                    questions.append(Question(field_key, field_title, cur_section, "states",
                                            control={"hint": "Select state..."}))
                    continue
                elif "date" in field_key.lower() or "dob" in field_key.lower() or field_key.lower() == "birth":
                    # Create a date field
                    questions.append(Question(field_key, field_title, cur_section, "date",
                                            control={"input_type": "past"}))
                    continue
                
                questions.append(Question(field_key, field_title, cur_section, "input",
                                          control={"input_type": input_type}))
            i += 1
            continue
        
        # Priority 2.1: Check for multi-field lines (e.g., "Phone: Mobile ___ Home ___ Work ___")
        # Improvement 3: Pass section and context for enhanced field naming
        prev_context = lines[max(0, i-3):i] if i > 0 else []
        multi_fields = detect_multi_field_line(line, cur_section, prev_context)
        if multi_fields:
            if debug:
                print(f"  [debug] multi-field detected: {line[:60]}... -> {len(multi_fields)} fields")
                if multi_fields:
                    print(f"  [debug]   Keys: {[k for k, _ in multi_fields]}")
            for field_key, field_title in multi_fields:
                # Determine input type based on base field
                input_type = "text"
                if "phone" in field_key or "fax" in field_key:
                    input_type = "phone"
                elif "email" in field_key:
                    input_type = "email"
                questions.append(Question(field_key, field_title, cur_section, "input",
                                          control={"input_type": input_type}))
            i += 1
            continue
        
        # Category 1 Fix 1.2: Check for fill-in-blank fields
        prev_line_text = lines[i-1] if i > 0 else None
        next_line_text = lines[i+1] if i+1 < len(lines) else None
        fill_in_blank = detect_fill_in_blank_field(line, prev_line_text, next_line_text)
        if fill_in_blank:
            field_key, field_title = fill_in_blank
            if debug:
                print(f"  [debug] fill-in-blank detected: {line[:60]}... -> {field_key}")
            
            # Determine field type based on the title
            if 'signature' in field_title.lower():
                questions.append(Question(field_key, field_title, cur_section, "block_signature",
                                        control={"language": "en", "variant": "adult_no_guardian_details"}))
            elif 'date' in field_title.lower():
                questions.append(Question(field_key, field_title, cur_section, "date",
                                        control={"input_type": "past"}))
            elif 'name' in field_title.lower():
                questions.append(Question(field_key, field_title, cur_section, "input",
                                        control={"hint": None, "input_type": "name"}))
            else:
                questions.append(Question(field_key, field_title, cur_section, "input",
                                        control={"input_type": "text"}))
            i += 1
            continue
        
        # Production Parity Fix: Check for embedded parenthetical fields
        # Detects patterns like "I, _____(print name) have been..."
        embedded_field = detect_embedded_parenthetical_field(line)
        if embedded_field:
            field_key, field_title = embedded_field
            if debug:
                print(f"  [debug] embedded parenthetical field: {line[:60]}... -> {field_key}")
            
            # Determine field type based on key
            if 'signature' in field_key:
                questions.append(Question(field_key, field_title, cur_section, "block_signature",
                                        control={"language": "en", "variant": "adult_no_guardian_details"}))
            elif 'date' in field_key:
                questions.append(Question(field_key, field_title, cur_section, "date",
                                        control={"input_type": "past"}))
            else:
                # Most commonly a name field
                input_type = "name" if 'name' in field_key else "text"
                questions.append(Question(field_key, field_title, cur_section, "input",
                                        control={"hint": None, "input_type": input_type}))
            i += 1
            continue
        
        # Priority 2.3: Check for inline checkbox with continuation text
        inline_checkbox_field = detect_inline_checkbox_with_text(line)
        if inline_checkbox_field:
            field_key, field_title, field_type = inline_checkbox_field
            if debug:
                print(f"  [debug] inline checkbox with text: {line[:60]}... -> {field_key}")
            # Create a boolean/radio field with the option and description
            questions.append(Question(field_key, field_title, cur_section, field_type,
                                      control={"options": [make_option("Yes", True), make_option("No", False)]}))
            i += 1
            continue

        # Simple labeled fields
        if STATE_LABEL_RE.match(title):
            key = slugify(title or "state")
            if insurance_scope and "insurance" in cur_section.lower(): key = f"{key}{insurance_scope}"
            questions.append(Question(key, title or "State", cur_section, "states", control={}))
            i += 1; continue

        # Production readiness: Remove trailing underscores before date detection
        title_for_check = re.sub(r'_+$', '', title).strip()
        if DATE_LABEL_RE.search(title_for_check):
            # Improvement 6: Use contextual date key instead of generic slugify
            prev_context = lines[max(0, i-3):i] if i > 0 else []
            key = generate_contextual_date_key(title or "date", prev_context, cur_section)
            
            if insurance_scope and "insurance" in cur_section.lower(): 
                key = f"{key}{insurance_scope}"
            # Archivev18 Fix 1: Clean date field titles to remove template artifacts
            clean_title = clean_field_title(title) if title else "Date"
            questions.append(Question(key, clean_title, cur_section, "date",
                                      control={"input_type": classify_date_input(title)}))
            i += 1; continue

        collected = opts_inline or opts_block
        if collected:
            # Fix 1: Clean title if it has inline checkboxes or concatenated options
            clean_title = title
            
            # If we have inline options (especially multiple ones), check if title needs cleaning
            if opts_inline and len(opts_inline) >= 2:
                # Archivev19 Fix 3: Only look back if current line doesn't have a clear label before checkboxes
                # Extract text before first checkbox to see if we have a meaningful title
                extracted_title = extract_title_from_inline_checkboxes(line)
                
                # Archivev20 Fix 3: If we extracted a title with proper formatting (has colon), ALWAYS use it
                # Don't look back even if it's short - this fixes "Marital Status:" being replaced by "Ext#"
                has_proper_label = extracted_title and ':' in line[:line.find('[') if '[' in line else len(line)]
                
                # Phase 4 Fix: Common short field names that should be accepted even without colon
                common_short_fields = {'sex', 'gender', 'age', 'dob', 'ssn', 'name', 'city', 'state', 'zip'}
                is_common_short = extracted_title and extracted_title.lower().strip() in common_short_fields
                
                # If we extracted a meaningful title from the current line, use it
                if extracted_title and len(extracted_title) >= 3 and (has_proper_label or is_common_short):
                    # Current line has "Label: [ ] options" format OR is a common short field - use the extracted label
                    clean_title = extracted_title
                elif extracted_title and len(extracted_title) >= 5:
                    # Extracted title is long enough even without colon
                    clean_title = extracted_title
                else:
                    # Title likely includes the options. Look back for a better title (skip blank lines).
                    lookback_idx = i - 1
                    while lookback_idx >= 0 and not lines[lookback_idx].strip():
                        lookback_idx -= 1  # Skip blank lines
                    if lookback_idx >= 0:
                        prev_line = collapse_spaced_caps(lines[lookback_idx].strip())
                        if prev_line and not re.search(CHECKBOX_ANY, prev_line) and not is_heading(prev_line):
                            # Use previous line if it looks like a question/prompt
                            if len(prev_line) >= 5:
                                clean_title = prev_line.rstrip(':').strip()
            
            # If title still has checkbox markers, try to extract clean text
            if re.search(CHECKBOX_ANY, clean_title):
                extracted = extract_title_from_inline_checkboxes(clean_title)
                # Archivev12 Fix: Allow short field names like "Sex", "Age"
                if extracted and len(extracted) >= 2:
                    clean_title = extracted
                else:
                    # Couldn't extract - fallback to looking back or generic title
                    if i > 0 and clean_title == title:  # Haven't already looked back
                        prev_line = collapse_spaced_caps(lines[i-1].strip())
                        if prev_line and not re.search(CHECKBOX_ANY, prev_line) and not is_heading(prev_line):
                            clean_title = prev_line.rstrip(':').strip()
                        else:
                            clean_title = "Please select"
                    else:
                        clean_title = "Please select"
            
            # Archivev21 Fix 7: Clean title before creating key to fix OCR errors in keys
            cleaned_question_title = clean_field_title(clean_title)
            
            lowset = {n.lower() for (n,_b) in collected}
            if {"yes","no"} <= lowset or lowset <= YESNO_SET:
                control = {"options":[make_option("Yes",True), make_option("No",False)]}
                if re.search(IF_GUIDANCE_RE, cleaned_question_title):
                    control["extra"] = {"type":"Input","value":True,"optional":True,"hint":"If yes, please explain"}
                    # --- Fix 2: Add separate input for "if yes" (enhanced) ---
                    if j < len(lines):
                        next_line = collapse_spaced_caps(lines[j].strip())
                        if re.search(r"\b(list|explain|if so|name of)\b", next_line, re.I):
                            follow_up_title = f"{cleaned_question_title} - Details"
                            follow_up_key = slugify(follow_up_title)
                            questions.append(Question(follow_up_key, follow_up_title, cur_section, "input", control={"input_type": "text"}))
                key = slugify(cleaned_question_title)
                if insurance_scope and "insurance" in cur_section.lower(): key = f"{key}{insurance_scope}"
                questions.append(Question(key, cleaned_question_title, cur_section, "radio", control=control))
            else:
                make_radio = bool(SINGLE_SELECT_TITLES_RE.search(cleaned_question_title))
                options = [make_option(n, b) for (n,b) in collected]
                if not options and ("," in cleaned_question_title or "/" in cleaned_question_title or ";" in cleaned_question_title):
                    for tok in re.split(r"[,/;]", cleaned_question_title):
                        tok = clean_token(tok)
                        if tok: options.append(make_option(tok, None))
                
                # NEW Improvement 8: Use smart multi-select detection
                # Override make_radio decision with context-aware logic
                option_names = [opt['name'] for opt in options]
                is_multi = infer_multi_select_from_context(cleaned_question_title, option_names, cur_section)
                make_radio = not is_multi  # If multi-select, not radio; if single-select, is radio
                
                control: Dict = {"options": options}
                if not make_radio:
                    control["multi"] = True
                if is_hear:
                    control["extra"] = {"type":"Input","value":True,"optional":True,"hint":"Other (please specify)"}
                    if (REFERRED_BY_RE.search(cleaned_question_title) or (j < len(lines) and REFERRED_BY_RE.search(lines[j]))):
                        questions.append(Question("referred_by","Referred by",cur_section,"input",control={"input_type":"text"}))
                key = slugify(cleaned_question_title)
                if insurance_scope and "insurance" in cur_section.lower(): key = f"{key}{insurance_scope}"
                qtype = "radio" if make_radio else "dropdown"
                questions.append(Question(key, cleaned_question_title, cur_section, qtype, control=control))
            i = j if not opts_inline else i + 1
            continue

        # Long paragraph → terms
        # First check if the current line is a known field label - if so, don't treat as paragraph
        current_line_is_field_label = False
        for field_key, pattern in KNOWN_FIELD_LABELS.items():
            if re.search(pattern, line, re.I):
                current_line_is_field_label = True
                break
        
        if not current_line_is_field_label:
            para = [lines[i]]; k = i+1
            while k < len(lines) and lines[k].strip() and not BULLET_RE.match(lines[k].strip()):
                if is_heading(lines[k]): break
                # Stop collecting if we hit a yes/no question pattern (don't include these in terms)
                if extract_compound_yn_prompts(lines[k]):
                    break
                # Don't absorb lines that look like field labels (e.g., "Patient Name:", "Date of Birth:")
                # Check if line matches known field label patterns
                line_matches_field_label = False
                for field_key, pattern in KNOWN_FIELD_LABELS.items():
                    if re.search(pattern, lines[k], re.I):
                        line_matches_field_label = True
                        break
                if line_matches_field_label:
                    break
                para.append(lines[k]); k += 1
            joined = " ".join(collapse_spaced_caps(x).strip() for x in para)
            if len(joined) > 250 and joined.count(".") >= 2:
                if debug:
                    print(f"  [debug] capturing long paragraph as terms: len={len(joined)}, periods={joined.count('.')}")
                chunks: List[List[str]] = []; cur: List[str] = []
                for s in para:
                    if is_heading(s.strip()) and cur:
                        chunks.append(cur); cur=[s]
                    else:
                        cur.append(s)
                if cur: chunks.append(cur)
                for idx2, chunk in enumerate(chunks):
                    t = " ".join(collapse_spaced_caps(x).strip() for x in chunk).strip()
                    if not t: continue
                    questions.append(Question(
                        slugify((chunk[0].strip() if is_heading(chunk[0].strip()) else (title or "terms")) + (f"_{idx2+1}" if idx2 else "")),
                        (collapse_spaced_caps(chunk[0].strip().rstrip(":")) if is_heading(chunk[0].strip()) else "Terms"),
                        "Consent",
                        "terms",
                        control={"agree_text":"I have read and agree to the terms.","html_text":t},
                    ))
                i = k; continue

        # Default: input
        # Fix 1: Skip if next line has inline options that will use this as title
        # BUT only if the next line doesn't have its own label (e.g., "Label: [ ] options")
        if i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if next_line and re.search(CHECKBOX_ANY, next_line):
                # Check if next line has inline options
                next_opts = options_from_inline_line(next_line)
                if len(next_opts) >= 2:
                    # Archivev20 Fix 4: Only skip if next line doesn't have its own label
                    # Check if next line has a colon before the checkboxes (indicates it has its own label)
                    bracket_pos = next_line.find('[')
                    has_own_label = bracket_pos > 0 and ':' in next_line[:bracket_pos]
                    
                    if not has_own_label:
                        # Next line will create a field using this line as title, so skip
                        i += 1
                        continue
        
        # Fix 2: Skip obvious business/practice address lines (Archivev20)
        # These should have been filtered by scrub_headers_footers but some slip through
        # Examples: "3138 N Lincoln Ave Chicago, IL", "60657", "Chicago, IL 60632"
        # Archivev22 Enhancement: Also skip document titles and section headers that slipped through
        # Archivev23 Enhancement: Skip sentence fragments, noise, and non-field questions
        skip_patterns = [
            r'^\d{5}$',  # Just a zip code
            r'^\d+\s+[NS]?\s*\w+\s+(Ave|Avenue|Rd|Road|St|Street|Blvd|Boulevard)\b',  # Street address
            r',\s*[A-Z]{2}\s+\d{5}$',  # City, State Zip
            r'^\d+[A-Z]?\s+S\s+\w+\s+(Ave|Rd|St|Blvd)',  # Address starting with number
            r'\w+\s*\.\s*(com|org|net|us|info|dental)\b',  # Website domain (with optional space before dot)
            r'^\d{3}[-.]\d{3}[-.]\d{4}$',  # Phone number (standalone)
            r'\b\d{3}[-.]\d{3}[-.]\d{4}\b',  # Phone number anywhere in text
            r'@\s*\w+\s*\.\s*\w+',  # Email address (with optional spaces)
            r'\(pg\.\s*\d+\)',  # Page number reference
            r'^please\s+(ensure|note|read|remember)',  # Instructions
            r'^[*]+',  # Lines starting with asterisks (often instructions)
            r'^[=|]+\s*\w{1,3}\s*[=|]+$',  # Symbol noise like "= ie |"
            r'^\([A-Z]{2,}\s+[A-Za-z\s]+\)$',  # Parenthetical codes like "(CF Gingivectomy)"
            r'^within\s+[-\d]+\s+(days?|hours?|weeks?)',  # Time references like "within -5 days"
        ]
        should_skip = any(re.search(pattern, title, re.I) for pattern in skip_patterns)
        
        # Parity Fix: Skip numbered consent section headings (e.g., "1. Reduction of tooth structure")
        # These are descriptive headings in consent forms, not fillable fields
        # Check regardless of current section, as section may not be set yet during parsing
        if not should_skip:
            # Pattern: starts with digit, period, space, then capitalized text
            # Examples: "1. Numbness following use of anesthesia", "3. Sensitivity of teeth"
            numbered_heading_match = re.match(r'^\d+\.\s+[A-Z]', title)
            if numbered_heading_match:
                # Additional check: verify this looks like a consent/informational heading
                # (not a numbered question like "1. Patient Name" which would have a colon or be very short)
                has_colon = ':' in title
                is_short = len(title.split()) <= 3
                # Skip if it's a longer numbered heading without colon (consent section pattern)
                if not has_colon and not is_short and len(title) > 10:
                    should_skip = True
                    if debug: print(f"  [debug] skipping numbered consent heading: '{title[:60]}'")
        
        # Parity Fix: Skip bullet-point risk descriptions in Consent sections
        # These are informational lists, not fields to be filled
        # Check regardless of section as consent patterns can appear anywhere
        if not should_skip:
            # Pattern: starts with bullet symbol (●, •, ·, -, *) followed by descriptive text
            # Common in risk/complication lists
            bullet_risk_match = re.match(r'^[●•·\-\*]\s+', title)
            if bullet_risk_match:
                # Check if it's a risk/complication description (not a checkbox option)
                risk_keywords = ['risk', 'complication', 'may', 'can', 'possible', 'potential',
                                'numbness', 'infection', 'swelling', 'pain', 'sensitivity',
                                'bleeding', 'reaction', 'damage', 'fracture', 'decay',
                                'temporary', 'permanent', 'post', 'treatment', 'procedure']
                has_risk_keywords = any(kw in title.lower() for kw in risk_keywords)
                # Also skip if it's a single word/short phrase (likely a condition name)
                is_short_item = len(title.split()) <= 4
                if has_risk_keywords or is_short_item:
                    should_skip = True
                    if debug: print(f"  [debug] skipping bullet risk description: '{title[:60]}'")
        
        # Archivev23 Enhancement: Skip sentence fragments that end with a period but aren't questions
        # (e.g., "healthy gums tissue.", "It is carried out to provide...")
        # Field labels typically DON'T end with periods unless they're questions or abbreviations
        if not should_skip and title.endswith('.'):
            # Check if this is NOT a question and NOT an abbreviation
            is_question = '?' in title
            is_abbreviation = len(title) <= 10 and title.count('.') == 1
            # Known field-like patterns that can end with period
            is_field_pattern = any(pattern in title.lower() for pattern in ['dr.', 'mr.', 'mrs.', 'ms.', 'inc.', 'ltd.'])
            
            if not is_question and not is_abbreviation and not is_field_pattern:
                # This looks like a sentence fragment or complete sentence, not a field label
                word_count = len(title.split())
                # Sentence fragments are typically 3+ words ending with period
                if word_count >= 3:
                    should_skip = True
                    if debug: print(f"  [debug] skipping sentence fragment: '{title[:60]}'")
        
        # Archivev23 Fix: Skip malformed merged field labels (extraction artifacts)
        # These have pattern: "Word1 Word2 Label:" indicating truncated first field + second field
        # Example: "Male Female Marital Status:" (Gender was already extracted)
        if not should_skip:
            # Check for pattern: words followed by capitalized Label followed by colon
            # But NOT if the first part looks like a complete field (e.g., has its own colon)
            words = title.split()
            if len(words) >= 3 and ':' in title:
                # Find position of last colon
                last_colon_pos = title.rfind(':')
                before_colon = title[:last_colon_pos].strip()
                words_before = before_colon.split()
                
                # If we have 2+ words before colon and none of them are common prefixes/articles
                # and the last 2 words look like a field label (both capitalized)
                if len(words_before) >= 2:
                    last_two = words_before[-2:]
                    # Check if last two words are capitalized (looks like "Marital Status")
                    if all(w and w[0].isupper() for w in last_two):
                        # Check if earlier words are option-like (Male, Female, etc.)
                        earlier_words = words_before[:-2]
                        looks_like_options = all(w and w[0].isupper() and len(w) <= 10 for w in earlier_words)
                        if looks_like_options and len(earlier_words) >= 1:
                            should_skip = True
                            if debug: print(f"  [debug] skipping merged field fragment: '{title[:60]}'")
        
        # Archivev23 Enhancement: Skip questions that are clearly section headings, not fields
        # (e.g., "What are the risks?" vs "Are you pregnant?" which IS a field)
        # Section heading questions typically ask about topics, not patient status
        if not should_skip and '?' in title:
            # Common section heading question patterns
            heading_question_patterns = [
                r'^what\s+(are|is)\s+the\s+(risks?|benefits?|alternatives?|procedures?|options?)',
                r'^how\s+(does|do|will|can)\s+(the|this|it)',
                r'^why\s+(is|do|does|would)',
                r'^when\s+(should|will|can)',
                r'^who\s+(should|will|can|is)',
            ]
            is_heading_question = any(re.match(pattern, title.lower()) for pattern in heading_question_patterns)
            
            if is_heading_question:
                should_skip = True
                if debug: print(f"  [debug] skipping section heading question: '{title[:60]}'")
        
        # Archivev22 Enhancement: Skip document titles that look like form headers
        # (e.g., "ENDODONTIC INFORMATION AND CONSENT FORM", "Informed Consent for Tooth Extraction")
        # Archivev23 Fix: Relax title case requirement - accept any capitalized title with form keywords
        # Parity Fix: Also handle 3-word titles like "Endodontic Informed Consent"
        if not should_skip and len(title.split()) >= 3:
            title_lower = title.lower()
            # Check for form title keywords
            has_form_keywords = any(kw in title_lower for kw in ['consent', 'form', 'information', 'agreement', 'authorization', 'release', 'disclosure'])
            # Check if it looks like a title (first letter capitalized, multiple capitalized words)
            words = title.split()
            capitalized_words = sum(1 for w in words if w and w[0].isupper())
            # Relaxed: 2+ capitalized words OR title case OR all caps
            looks_like_title = capitalized_words >= 2 or title.istitle() or title.isupper()
            
            # For 3-word titles, require stronger signal (e.g., "Informed Consent")
            if len(words) == 3:
                # Common 3-word consent patterns
                consent_patterns = ['informed consent', 'consent form', 'patient consent', 'consent agreement']
                has_consent_pattern = any(pattern in title_lower for pattern in consent_patterns)
                if has_form_keywords and has_consent_pattern and looks_like_title:
                    should_skip = True
                    if debug: print(f"  [debug] skipping document title: '{title[:60]}'")
            elif len(words) >= 4:
                # 4+ word titles with form keywords
                if has_form_keywords and looks_like_title:
                    should_skip = True
                    if debug: print(f"  [debug] skipping document title: '{title[:60]}'")
        
        # Archivev22 Enhancement: Skip lines that look like section headers describing content
        # (e.g., "Endodontic (Root Canal) Treatment, Endodontic Surgery, Anesthetics, and Medications")
        if not should_skip and len(title) > 60 and title.count(',') >= 2:
            # Long lines with multiple commas are likely descriptive headers, not field labels
            should_skip = True
            if debug: print(f"  [debug] skipping descriptive header: '{title[:60]}'")
        
        if should_skip:
            i += 1
            continue
        
        # Archivev21 Fix 2: Clean field title before creating Question (removes underscores, artifacts)
        cleaned_title = clean_field_title(title)
        itype = classify_input_type(cleaned_title)
        key = slugify(cleaned_title)
        # parent/guardian routing
        key, qtype, ctrl = _emit_parent_guardian_override(cleaned_title, key, "input", {"input_type": itype} if itype else {}, cur_section, insurance_scope, debug)
        # employer patient-first: only map to insurance_employer if insurance context tokens are present
        if key == "employer":
            if "insurance" in cur_section.lower() or re.search(r"\b(insured|subscriber|policy|member|insurance)\b", cleaned_title.lower()):
                pass  # allow later dictionary to map to insurance_employer if template title says so
            else:
                if debug: print(f"  [debug] gate: employer_patient_first -> '{cleaned_title}' -> employer (patient)")
        key = _insurance_scope_key(key, cur_section, insurance_scope, cleaned_title, debug)
        if insurance_scope and "insurance" in cur_section.lower() and not key.endswith((PRIMARY_SUFFIX, SECONDARY_SUFFIX)) and key in {"ssn","insurance_id_number"}:
            key = f"{key}{insurance_scope}"
        questions.append(Question(key, cleaned_title, cur_section, qtype, control=ctrl))
        i += 1

    # Final sanitation & signature rule
    def is_instructional_text(text: str) -> bool:
        """
        Archivev18 Fix 2: Detect instructional/paragraph text that shouldn't be captured as fields.
        Archivev22 Enhancement: Better detection of informational bullet points and long descriptions.
        Returns True if the text is likely instructions/guidance rather than a field label.
        """
        text_lower = text.lower().strip()
        original_text = text.strip()
        
        # Check if this is an informational bullet point (starts with bullet symbol)
        # Common in consent forms describing risks/complications
        if original_text.startswith(('●', '•', '·')):
            # These are typically risk descriptions, not fillable fields
            # Look for keywords common in risk/complication lists
            risk_keywords = [
                'adverse', 'reaction', 'numbness', 'tingling', 'bleeding', 'infection',
                'swelling', 'fracture', 'damage', 'injury', 'pain', 'sensitivity',
                'complication', 'risk', 'temporary', 'permanent', 'may result', 'may cause',
                'can lead', 'potential', 'possible', 'delayed healing', 'post-operative',
                'post treatment', 'involving', 'involvement'
            ]
            if any(keyword in text_lower for keyword in risk_keywords):
                return True
        
        # Check for common instructional phrases
        instructional_phrases = [
            "thank you for",
            "medication that you may be taking",
            "could have an important",
            "interrelationship with",
            "health problems that you may have",
            "please read",
            "please ensure",
            "please note",
            "i understand that providing incorrect",
            "to the best of my knowledge",
            "the questions on this form have been",
            "accurately answered",
            "providing incorrect information can be",
            "although dental professionals",
            "your mouth is a part of",
            "have been explained to me",
            "i understand that",
            "these treatments include",
            "alternative methods",
            "risks and complications",
            "may include (but",
            "are not limited to",
            "has performed a thorough examination",
            "has determined that",
        ]
        
        if any(phrase in text_lower for phrase in instructional_phrases):
            return True
        
        # If text is very long (>120 chars) and contains connecting phrases, likely instructional
        if len(text) > 120:
            connecting_phrases = [" that you ", " which ", " could ", " should ", " would ", " may ", " can "]
            if any(phrase in text_lower for phrase in connecting_phrases):
                return True
        
        # Check for document titles (all caps or title case, contains form keywords)
        if len(original_text.split()) >= 4:
            is_title_case = original_text.istitle() or original_text.isupper()
            title_keywords = ['consent', 'form', 'information', 'agreement', 'authorization', 'release', 'disclosure']
            has_form_keywords = any(keyword in text_lower for keyword in title_keywords)
            
            # Also check for common section header patterns
            section_keywords = ['risks', 'benefits', 'alternatives', 'procedures', 'treatment']
            has_section_keywords = any(keyword in text_lower for keyword in section_keywords)
            
            if is_title_case and (has_form_keywords or has_section_keywords):
                return True
        
        # Phase 4 Fix 10: Enhanced document title detection
        # Catch patterns like "Informed Consent for [Procedure Name]"
        if text_lower.startswith('informed consent for'):
            return True
        
        return False
    
    def is_junk(q: Question) -> bool:
        t = q.title.strip().lower()
        return (q.key == "q" or len(q.title.strip()) <= 1 or
                t in {"<<<", ">>>", "-", "—", "continued on back side"} or
                is_instructional_text(q.title))
    questions = [q for q in questions if not is_junk(q)]
    questions = [q for q in questions if not WITNESS_RE.search(q.title)]

    sig_idxs = [idx for idx,q in enumerate(questions) if q.type=="signature"]
    if not sig_idxs:
        questions.append(Question("signature","Signature","Signature","signature"))
    elif len(sig_idxs) > 1:
        for idx in reversed(sig_idxs[1:]): questions.pop(idx)

    return questions

# ============================================================================
# SECTION 4: VALIDATION AND DEDUPLICATION
# ============================================================================
# Functions for ensuring data quality and removing duplicates.
# Future PR: Move to modules/validation.py (~100 lines)

# ---------- Validation / normalization

def ensure_control_present(q: Question) -> None:
    if q.control is None: q.control = {}
    if q.type == "radio":    q.control.setdefault("options", [])
    if q.type == "dropdown": q.control.setdefault("options", []); q.control.setdefault("multi", True)
    if q.type == "date":     q.control.setdefault("input_type", "past")
    if q.type == "input":    q.control.setdefault("input_type", "text")
    if q.type == "terms":
        q.control.setdefault("agree_text","I have read and agree to the terms.")
        q.control.setdefault("html_text","")

def fill_missing_option_values(q: Question) -> None:
    if q.type not in ("radio","dropdown"): return
    opts = q.control.get("options") or []
    fixed=[]
    for opt in opts:
        name = (opt.get("name") or "").strip() or "Option"
        val  = opt.get("value")
        if val in (None, ""):
            vlow = name.strip().lower()
            if vlow == "yes": val = True
            elif vlow == "no": val = False
            else: val = slugify(name, 80)
        fixed.append({"name": name, "value": val})
    q.control["options"] = fixed

def _semantic_dedupe(payload: List[dict]) -> List[dict]:
    """
    Remove duplicate simple inputs that are identical semantically (same title/section/type/input_type).
    
    Archivev8 Fix 3: Skip deduplication for conditional fields (they can have same title).
    """
    seen: Set[Tuple[str,str,str,str]] = set()
    out: List[dict] = []
    for q in payload:
        if q.get("type") == "input":
            # Archivev8 Fix 3: Don't dedupe conditional fields
            if q.get("if") or "conditional" in q.get("key", "").lower() or "_explanation" in q.get("key", "").lower():
                out.append(q)
                continue
            
            itype = (q.get("control") or {}).get("input_type","text")
            sig = (q.get("title","").strip().lower(), q.get("section",""), "input", itype)
            if sig in seen:
                continue
            seen.add(sig)
        out.append(q)
    return out

def dedupe_keys(questions: List[Question]) -> None:
    """
    Deduplicate keys across all questions, including nested questions in multiradio controls.
    Ensures global uniqueness as required by Modento schema.
    """
    seen: Dict[str,int] = {}
    
    def dedupe_question(q: Question) -> None:
        """Deduplicate a single question and its nested questions recursively."""
        base = "signature" if q.type=="signature" else (q.key or "q")
        if q.type=="signature":
            q.key = "signature"
        elif base not in seen:
            seen[base]=1
            q.key=base
        else:
            seen[base]+=1
            q.key=f"{base}_{seen[base]}"
        
        # Handle nested questions in multiradio controls
        if q.type == "multiradio" and q.control and "questions" in q.control:
            nested_questions = q.control.get("questions", [])
            for nested_q in nested_questions:
                # Process nested question if it's a Question object
                if isinstance(nested_q, Question):
                    dedupe_question(nested_q)
                # Process nested question if it's a dict
                elif isinstance(nested_q, dict):
                    nested_key = nested_q.get("key", "q")
                    if nested_key in seen:
                        seen[nested_key] += 1
                        nested_q["key"] = f"{nested_key}_{seen[nested_key]}"
                    else:
                        seen[nested_key] = 1
    
    for q in questions:
        dedupe_question(q)

def validate_form(questions: List[Question]) -> List[str]:
    errs=[]
    for q in questions:
        if q.type not in ALLOWED_TYPES:
            errs.append(f"Unsupported type for {q.key}: {q.type}")
    sig = [q for q in questions if q.type=="signature"]
    if len(sig)!=1 or sig[0].key!="signature":
        errs.append("Signature rule violated (need exactly one with key='signature').")
    for q in questions:
        if q.type in ("radio","dropdown","checkbox"):
            for opt in q.control.get("options", []):
                if opt.get("value") in (None,""):
                    errs.append(f"Empty option value in {q.key}")
        # Patch 2: Validate field key format for Modento compliance
        if not is_valid_modento_key(q.key):
            errs.append(f"Invalid key format: '{q.key}' (must be snake_case with only lowercase letters, digits, underscores)")
    return errs


def is_valid_modento_key(key: str) -> bool:
    """
    Validate that a field key conforms to Modento's expected format.
    
    Patch 2: Field Key Validation
    - Keys must be snake_case (lowercase letters, digits, underscores only)
    - Must not start with a digit (enforced by slugify, but checked here too)
    - Must not be empty
    - No uppercase letters, hyphens, or other special characters allowed
    
    Args:
        key: The field key to validate
        
    Returns:
        True if the key is valid, False otherwise
        
    Examples:
        >>> is_valid_modento_key("patient_name")
        True
        >>> is_valid_modento_key("date_of_birth")
        True
        >>> is_valid_modento_key("phone_1")
        True
        >>> is_valid_modento_key("Patient-Name")  # uppercase/hyphen
        False
        >>> is_valid_modento_key("123_start")  # starts with digit
        False
        >>> is_valid_modento_key("")  # empty
        False
    """
    if not key:
        return False
    
    # Must match snake_case pattern: lowercase letters, digits, underscores
    # Must not start with a digit
    pattern = r'^[a-z_][a-z0-9_]*$'
    return bool(re.match(pattern, key))

# ============================================================================
# SECTION 5: POSTPROCESSING FUNCTIONS
# ============================================================================
# Functions for consolidating, merging, and cleaning the parsed field data.
# Future PR: Move to modules/postprocessing.py (~770 lines)

def questions_to_json(questions: List[Question]) -> List[Dict]:
    for q in questions:
        ensure_control_present(q)
        fill_missing_option_values(q)
    dedupe_keys(questions)
    errs = validate_form(questions)
    if errs:
        print("Validation warnings:", *errs, sep="\n  ", file=sys.stderr)
    payload = []
    for q in questions:
        item = {"key": q.key, "title": q.title, "section": q.section,
                "optional": q.optional, "type": q.type, "control": q.control or {}}
        
        # Fix 2: Add conditional "if" property for follow-up fields
        if hasattr(q, 'conditional_on') and q.conditional_on:
            # Convert conditional_on to "if" array format
            item["if"] = [{"key": cond_key, "value": cond_val} for cond_key, cond_val in q.conditional_on]
        
        payload.append(item)
    
    payload = _semantic_dedupe(payload)
    return payload

# ---------- Post-processing helpers

def postprocess_merge_hear_about_us(payload: List[dict]) -> List[dict]:
    idxs = [i for i,q in enumerate(payload) if HEAR_ABOUT_RE.search(q.get("title",""))]
    if len(idxs) <= 1:
        return payload
    keep = idxs[0]
    all_opts: Dict[str, dict] = {}
    extras: List[dict] = []
    for i in idxs:
        q = payload[i]
        for o in (q.get("control",{}).get("options") or []):
            k = normalize_opt_name(o.get("name",""))
            if not k: continue
            all_opts[k] = {"name": o.get("name",""), "value": o.get("value", slugify(o.get("name",""),80))}
        extra = q.get("control",{}).get("extra")
        if extra: extras.append(extra)
    payload[keep]["control"]["options"] = list(all_opts.values())
    payload[keep]["control"]["multi"] = True
    if extras:
        payload[keep]["control"]["extra"] = {"type":"Input","hint":"Other (please specify)"}
    for i in sorted(idxs[1:], reverse=True):
        payload.pop(i)
    return payload

# ---------- Fix 3: Enhanced Malformed Conditions Detection

def is_malformed_condition_field(field: dict) -> bool:
    """
    Detect if a field is a malformed medical/dental condition field.
    
    Criteria:
    - Type is dropdown with multi=True
    - Title is unusually long (5+ words) or contains multiple condition keywords
    - Options contain medical/health terms
    
    Examples of malformed titles:
    - "Artificial Angina (chest Heart pain) Valve Thyroid Disease..."
    - "Bleeding, Swollen, Irritated gums Tobacco"
    - "Heart Surgery Ulcers (Stomach) Dizziness AIDS"
    """
    if field.get('type') != 'dropdown':
        return False
    
    if not field.get('control', {}).get('multi'):
        return False
    
    title = field.get('title', '')
    title_lower = title.lower()
    
    # Check 1: Title has multiple medical condition keywords
    condition_keywords = [
        'diabetes', 'cancer', 'heart', 'disease', 'arthritis', 'hepatitis',
        'asthma', 'anxiety', 'depression', 'ulcer', 'thyroid', 'kidney',
        'liver', 'tuberculosis', 'hiv', 'aids', 'stroke', 'bleeding',
        'anemia', 'glaucoma', 'angina', 'valve', 'neurological'
    ]
    
    keyword_count = sum(1 for kw in condition_keywords if kw in title_lower)
    
    # If title has 3+ condition keywords, likely malformed
    if keyword_count >= 3:
        return True
    
    # Check 2: Title is very long and has some condition keywords
    word_count = len(title.split())
    if word_count >= 8 and keyword_count >= 2:
        return True
    
    # Check 3: Title contains multiple capitalized words that look like conditions
    capitalized_words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', title)
    if len(capitalized_words) >= 4 and keyword_count >= 1:
        return True
    
    return False


def _looks_like_medical_condition(opt_name: str) -> bool:
    """
    Helper function to check if an option name looks like a medical condition.
    Used by postprocess_consolidate_medical_conditions.
    """
    w = norm_title(opt_name)
    return any(t in w for t in _COND_TOKENS)


def postprocess_consolidate_medical_conditions(payload: List[dict]) -> List[dict]:
    """
    Enhanced version that consolidates both well-formed and malformed condition dropdowns,
    plus individual checkbox/radio fields that look like medical conditions (Fix 1).
    """
    
    # Medical condition keywords for identifying individual condition fields
    CONDITION_KEYWORDS = [
        'diabetes', 'cancer', 'heart', 'disease', 'arthritis', 'hepatitis',
        'asthma', 'anxiety', 'depression', 'ulcer', 'thyroid', 'kidney',
        'liver', 'tuberculosis', 'hiv', 'aids', 'stroke', 'bleeding',
        'anemia', 'glaucoma', 'angina', 'valve', 'neurological', 'alzheimer',
        'blood pressure', 'cholesterol', 'pacemaker', 'chemotherapy', 'radiation',
        'convulsion', 'seizure', 'epilepsy', 'migraine', 'allergy'
    ]
    
    # Separate handling for malformed dropdowns, well-formed dropdowns, and individual checkboxes
    malformed_indices = []
    wellformed_groups_by_section: Dict[str, List[int]] = defaultdict(list)
    individual_condition_indices: Dict[str, List[int]] = defaultdict(list)
    
    for i, q in enumerate(payload):
        section = q.get('section', 'General')
        
        # Check if malformed
        if is_malformed_condition_field(q):
            malformed_indices.append(i)
            continue
        
        # Check if well-formed medical condition field (original logic)
        if (q.get('type') == 'dropdown' and 
            q.get('control', {}).get('multi', False) and 
            q.get('section') in {'Medical History', 'Dental History'}):
            
            opts = q.get('control', {}).get('options') or []
            if len(opts) >= 5 and sum(_looks_like_medical_condition(o.get('name', '')) for o in opts) >= 3:
                wellformed_groups_by_section[section].append(i)
                continue
        
        # Check if single-option multi-select dropdown (likely should be consolidated)
        if (q.get('type') == 'dropdown' and 
            q.get('control', {}).get('multi', False) and 
            section in {'Medical History', 'General', 'Dental History', 'Patient Information'}):
            
            opts = q.get('control', {}).get('options') or []
            if len(opts) == 1:
                # Single option dropdown - treat like individual condition
                title = opts[0].get('name', '').lower()
                has_condition_keyword = any(kw in title for kw in CONDITION_KEYWORDS)
                # Also check the field title
                field_title = q.get('title', '').lower()
                has_condition_in_title = any(kw in field_title for kw in CONDITION_KEYWORDS)
                
                if has_condition_keyword or has_condition_in_title:
                    individual_condition_indices[section].append(i)
                    continue
        
        # Check if individual checkbox/radio/input that looks like a medical condition (Fix 1)
        if q.get('type') in ['checkbox', 'radio', 'input'] and section in {'Medical History', 'General', 'Dental History'}:
            title = q.get('title', '').lower()
            # Check if title contains medical condition keywords
            has_condition_keyword = any(kw in title for kw in CONDITION_KEYWORDS)
            # Or if it's a short title (1-4 words) in Medical History/Dental History section
            is_short_medical = section in {'Medical History', 'Dental History'} and len(title.split()) <= 4
            
            if has_condition_keyword or is_short_medical:
                individual_condition_indices[section].append(i)
    
    # Consolidate malformed fields
    if malformed_indices:
        # Extract all options from malformed fields
        all_options = []
        sections = set()
        
        for idx in malformed_indices:
            field = payload[idx]
            sections.add(field.get('section', 'Medical History'))
            
            # Extract options from the malformed field
            opts = field.get('control', {}).get('options', [])
            for opt in opts:
                opt_name = opt.get('name', '')
                opt_value = opt.get('value', '')
                
                # Clean up option name
                opt_name = opt_name.strip()
                
                # Skip if too short or looks like junk
                if len(opt_name) < 3:
                    continue
                
                # Add to consolidated list if not duplicate
                if opt_name not in [o['name'] for o in all_options]:
                    all_options.append({
                        'name': opt_name,
                        'value': opt_value if opt_value else slugify(opt_name, 80)
                    })
        
        # Create consolidated field
        if all_options:
            consolidated_section = list(sections)[0] if len(sections) == 1 else 'Medical History'
            
            consolidated_field = {
                'key': 'medical_conditions_consolidated',
                'type': 'dropdown',
                'title': 'Do you have or have you had any of the following medical conditions?',
                'section': consolidated_section,
                'optional': False,
                'control': {
                    'options': sorted(all_options, key=lambda x: x['name']),
                    'multi': True
                }
            }
            
            # Replace first malformed field with consolidated
            payload[malformed_indices[0]] = consolidated_field
            
            # Remove other malformed fields
            for idx in sorted(malformed_indices[1:], reverse=True):
                payload.pop(idx)
    
    # Original consolidation logic for well-formed fields
    for section, groups in list(wellformed_groups_by_section.items()):
        if len(groups) <= 1:
            continue
        
        keep = groups[0]
        seen: Set[str] = set()
        merged: List[dict] = []
        
        for i in groups:
            for o in (payload[i].get('control', {}).get('options') or []):
                name = SPELL_FIX.get(o.get('name', ''), o.get('name', ''))
                norm = normalize_opt_name(name)
                if not norm or norm in seen:
                    continue
                seen.add(norm)
                merged.append({'name': name, 'value': o.get('value', slugify(name, 80))})
        
        if section == 'Medical History':
            payload[keep]['title'] = 'Medical Conditions'
            payload[keep]['key'] = 'medical_conditions'
        payload[keep]['control']['options'] = merged
        
        for i in sorted(groups[1:], reverse=True):
            payload.pop(i)
    
    # New: Consolidate individual checkbox/radio fields that are medical conditions (Fix 1)
    for section, indices in list(individual_condition_indices.items()):
        # Only consolidate if we have 5+ individual condition fields
        if len(indices) < 5:
            continue
        
        # Create consolidated dropdown
        consolidated_options = []
        seen_names: Set[str] = set()
        
        # Filter out invalid indices before processing
        valid_process_indices = [idx for idx in indices if idx < len(payload)]
        
        for idx in valid_process_indices:
            field = payload[idx]
            
            # For single-option dropdowns, use the option name
            if (field.get('type') == 'dropdown' and 
                field.get('control', {}).get('multi', False)):
                opts = field.get('control', {}).get('options', [])
                if len(opts) == 1:
                    title = opts[0].get('name', '').strip()
                else:
                    title = field.get('title', '').strip()
            else:
                title = field.get('title', '').strip()
            
            # Normalize the title for duplicate detection
            norm_title = normalize_opt_name(title)
            if norm_title and norm_title not in seen_names:
                seen_names.add(norm_title)
                consolidated_options.append({
                    'name': title,
                    'value': slugify(title, 80)
                })
        
        if consolidated_options and indices:
            # Verify indices are still valid after earlier consolidations
            valid_indices = [idx for idx in indices if idx < len(payload)]
            if not valid_indices:
                continue
                
            # Replace first field with consolidated dropdown
            payload[valid_indices[0]] = {
                'key': 'medical_conditions' if section == 'Medical History' else 'dental_conditions',
                'type': 'dropdown',
                'title': 'Do you have or have you had any of the following?',
                'section': section,
                'optional': False,
                'control': {
                    'options': sorted(consolidated_options, key=lambda x: x['name']),
                    'multi': True
                }
            }
            
            # Remove other individual condition fields
            for idx in sorted(valid_indices[1:], reverse=True):
                if idx < len(payload):
                    payload.pop(idx)
    
    return payload

def postprocess_signature_uniqueness(payload: List[dict]) -> List[dict]:
    sig_idx = [i for i,q in enumerate(payload) if q.get("type")=="signature"]
    if not sig_idx:
        payload.append({"key":"signature","title":"Signature","section":"Signature","type":"signature","control":{}})
    elif len(sig_idx) > 1:
        for i in sorted(sig_idx[1:], reverse=True):
            payload.pop(i)
        payload[sig_idx[0]].update({"key":"signature","title":"Signature","section":"Signature","type":"signature"})
    else:
        i = sig_idx[0]
        payload[i].update({"key":"signature","title":"Signature","section":"Signature","type":"signature"})
    return payload

def postprocess_rehome_by_key(payload: List[dict], dbg: Optional[DebugLogger]=None) -> List[dict]:
    ins_keys = ("insurance_", "insured", "policy", "group", "member id", "id number")
    demo_keys = ("last_name","first_name","mi","date_of_birth","address","city","state","zipcode","email","phone",
                 "mobile_phone","home_phone","work_phone","drivers_license","employer","occupation","parent_")
    for q in payload:
        t = norm_title(q.get("title",""))
        k = q.get("key","")
        sec = q.get("section","General")
        if (sec not in {"Insurance"} and (t.count("insurance") or any(s in k for s in ins_keys))):
            q["section"] = "Insurance"
            if dbg: dbg.gate(f"rehome -> Insurance :: {q.get('title','')}")
        if (sec not in {"Patient Information","Insurance"} and (any(k.startswith(d) for d in demo_keys) or "parent" in k)):
            q["section"] = "Patient Information"
            if dbg: dbg.gate(f"rehome -> Patient Information :: {q.get('title','')}")
    return payload

def postprocess_infer_sections(payload: List[dict], dbg: Optional[DebugLogger] = None) -> List[dict]:
    """
    Reassign fields from 'General' to more specific sections based on content.
    Uses keyword matching to identify medical and dental questions.
    
    Parity Improvement #12: Enhanced section detection with expanded keywords.
    """
    # Strong keywords that alone indicate medical history
    STRONG_MEDICAL_KEYWORDS = [
        'physician', 'hospitalized', 'surgery', 'surgical', 'operation',
        'medication', 'medicine', 'prescription', 
        'allergy', 'allergic',
        # Common disease/condition patterns
        'hiv', 'aids', 'diabetes', 'cancer', 'heart', 'blood pressure',
        'hepatitis', 'asthma', 'arthritis', 'alzheimer', 'anemia'
    ]
    
    MEDICAL_KEYWORDS = [
        'doctor', 'hospital', 
        'drug', 'pills',
        'illness', 'disease', 'condition', 'diagnosis',
        'reaction', 'symptom', 'discomfort', 'health',
        'care now', 'taking any', 'have you had', 'have you ever'
    ]
    
    DENTAL_KEYWORDS = [
        'tooth', 'teeth', 'gum', 'gums',
        'dental', 'dentist', 'orthodontic', 'orthodontist',
        'cleaning', 'cavity', 'cavities', 'crown', 'filling',
        'bite', 'jaw', 'tmj', 'smile'
    ]
    
    # Parity Improvement #12: Additional section keywords
    PATIENT_INFO_KEYWORDS = [
        'name', 'address', 'phone', 'email', 'birth', 'dob',
        'ssn', 'social security', 'gender', 'marital', 'employer',
        'occupation', 'contact', 'nickname', 'preferred name'
    ]
    
    INSURANCE_KEYWORDS = [
        'insurance', 'policy', 'coverage', 'carrier', 'subscriber',
        'member id', 'group number', 'plan', 'benefits', 'primary insurance',
        'secondary insurance', 'insurance company'
    ]
    
    EMERGENCY_KEYWORDS = [
        'emergency contact', 'emergency', 'notify', 'relationship',
        'emergency phone', 'in case of emergency'
    ]
    
    CONSENT_KEYWORDS = [
        'consent', 'acknowledge', 'agree', 'understand', 'authorize',
        'risks', 'complications', 'terms', 'conditions', 'authorization'
    ]
    
    for item in payload:
        if item.get('section') == 'General':
            title_lower = item.get('title', '').lower()
            key_lower = item.get('key', '').lower()
            combined = title_lower + ' ' + key_lower
            
            # Check for strong medical keywords (single match is enough)
            has_strong_medical = any(kw in combined for kw in STRONG_MEDICAL_KEYWORDS)
            
            # Count keyword matches for all categories
            medical_score = sum(1 for kw in MEDICAL_KEYWORDS if kw in combined)
            dental_score = sum(1 for kw in DENTAL_KEYWORDS if kw in combined)
            patient_score = sum(1 for kw in PATIENT_INFO_KEYWORDS if kw in combined)
            insurance_score = sum(1 for kw in INSURANCE_KEYWORDS if kw in combined)
            emergency_score = sum(1 for kw in EMERGENCY_KEYWORDS if kw in combined)
            consent_score = sum(1 for kw in CONSENT_KEYWORDS if kw in combined)
            
            # Reassign based on signals (in order of priority)
            # Patient Information
            if patient_score >= 2:
                if dbg:
                    dbg.gate(f"section_inference -> Patient Information :: {item.get('title', '')} (score={patient_score})")
                item['section'] = 'Patient Information'
            # Insurance
            elif insurance_score >= 1:
                if dbg:
                    dbg.gate(f"section_inference -> Insurance :: {item.get('title', '')} (score={insurance_score})")
                item['section'] = 'Insurance'
            # Emergency Contact
            elif emergency_score >= 1:
                if dbg:
                    dbg.gate(f"section_inference -> Emergency Contact :: {item.get('title', '')} (score={emergency_score})")
                item['section'] = 'Emergency Contact'
            # Consent/Terms
            elif consent_score >= 2 or (consent_score >= 1 and len(title_lower) > 100):
                if dbg:
                    dbg.gate(f"section_inference -> Consent :: {item.get('title', '')} (score={consent_score})")
                item['section'] = 'Consent'
            # Medical History (strong keyword alone is enough)
            elif has_strong_medical and dental_score == 0:
                if dbg:
                    dbg.gate(f"section_inference -> Medical History :: {item.get('title', '')} (strong keyword match)")
                item['section'] = 'Medical History'
            # Or 2+ regular medical keywords
            elif medical_score >= 2 and medical_score > dental_score:
                if dbg:
                    dbg.gate(f"section_inference -> Medical History :: {item.get('title', '')} (score={medical_score})")
                item['section'] = 'Medical History'
            # Dental History
            elif dental_score >= 2 and dental_score > medical_score:
                if dbg:
                    dbg.gate(f"section_inference -> Dental History :: {item.get('title', '')} (score={dental_score})")
                item['section'] = 'Dental History'
    
    return payload

def postprocess_filter_document_titles(payload: List[dict], dbg: Optional[DebugLogger] = None) -> List[dict]:
    """
    Parity Fix: Remove fields that are actually document titles.
    
    Document titles like "Endodontic Informed Consent" or "Extraction Consent"
    sometimes slip through parsing and get created as fields. Filter them out.
    """
    filtered = []
    removed_count = 0
    
    for field in payload:
        title = field.get('title', '')
        field_type = field.get('type', '')
        
        # Skip if not an input/checkbox field (terms, signatures are OK)
        if field_type not in ['input', 'checkbox']:
            filtered.append(field)
            continue
        
        title_lower = title.lower()
        words = title.split()
        
        # Pattern 1: Single word + "consent" (e.g., "extraction consent", "implant consent")
        if len(words) == 2 and 'consent' in title_lower:
            # Check if it's not a field-like pattern (no colon, no underscores)
            if ':' not in title and '_' not in title:
                removed_count += 1
                if dbg:
                    dbg.log(f"filter_document_titles -> Removed '{title}' (2-word consent title)")
                continue
        
        # Pattern 2: 3-word consent patterns (e.g., "Endodontic Informed Consent")
        if len(words) == 3:
            consent_patterns = ['informed consent', 'consent form', 'patient consent', 'consent agreement']
            if any(pattern in title_lower for pattern in consent_patterns):
                removed_count += 1
                if dbg:
                    dbg.log(f"filter_document_titles -> Removed '{title}' (3-word consent pattern)")
                continue
        
        # Pattern 3: 4+ word titles with "consent" or "form" keywords
        if len(words) >= 4:
            form_keywords = ['consent', 'form', 'information', 'agreement', 'authorization', 'release']
            if any(kw in title_lower for kw in form_keywords):
                # Check if it looks like a title (mostly capitalized)
                capitalized = sum(1 for w in words if w and w[0].isupper())
                if capitalized >= len(words) - 1:  # Allow one lowercase word
                    removed_count += 1
                    if dbg:
                        dbg.log(f"filter_document_titles -> Removed '{title}' (multi-word form title)")
                    continue
        
        filtered.append(field)
    
    if removed_count > 0 and dbg:
        dbg.log(f"Filtered {removed_count} document title fields")
    
    return filtered


def postprocess_consolidate_duplicates(payload: List[dict], dbg: Optional[DebugLogger] = None) -> List[dict]:
    """
    Remove duplicate fields, keeping the one in the most appropriate section.
    Focuses on common fields like DOB, phone, email, address, SSN.
    """
    # Common fields that might be duplicated with their preferred section
    COMMON_FIELDS = {
        'date_of_birth': 'Patient Information',
        'dob': 'Patient Information',
        'phone': 'Patient Information',
        'mobile_phone': 'Patient Information',
        'home_phone': 'Patient Information',
        'work_phone': 'Patient Information',
        'cell_phone': 'Patient Information',
        'email': 'Patient Information',
        'address': 'Patient Information',
        'ssn': 'Patient Information',
        'social_security': 'Patient Information',
    }
    
    # Track seen keys and their indices
    seen_keys = {}
    to_remove = []
    
    for i, item in enumerate(payload):
        key = item.get('key', '')
        # Normalize key by removing numeric suffixes and scope markers
        key_base = re.sub(r'_\d+$', '', key)  # Remove _2, _3, etc.
        key_base = re.sub(r'__\w+$', '', key_base)  # Remove __primary, __secondary, etc.
        
        # Check if this is a common field that might be duplicated
        if key_base in COMMON_FIELDS:
            if key_base in seen_keys:
                # Duplicate found - determine which to keep
                prev_idx = seen_keys[key_base]
                prev_section = payload[prev_idx].get('section', 'General')
                curr_section = item.get('section', 'General')
                preferred_section = COMMON_FIELDS[key_base]
                
                if curr_section == preferred_section and prev_section != preferred_section:
                    # Current is in preferred section, remove previous
                    to_remove.append(prev_idx)
                    seen_keys[key_base] = i
                    if dbg:
                        dbg.gate(f"duplicate_consolidation -> Removed {key_base} from {prev_section}, kept in {curr_section}")
                elif prev_section == preferred_section and curr_section != preferred_section:
                    # Previous is in preferred section, remove current
                    to_remove.append(i)
                    if dbg:
                        dbg.gate(f"duplicate_consolidation -> Removed {key} from {curr_section}, kept {key_base} in {prev_section}")
                else:
                    # Neither in preferred or both in preferred, keep first occurrence
                    to_remove.append(i)
                    if dbg:
                        dbg.gate(f"duplicate_consolidation -> Removed duplicate {key} from {curr_section}")
            else:
                seen_keys[key_base] = i
    
    # Remove duplicates in reverse order to maintain indices
    for idx in sorted(set(to_remove), reverse=True):
        payload.pop(idx)
    
    return payload


def postprocess_consolidate_malformed_grids(payload: List[dict], dbg: Optional[DebugLogger] = None) -> List[dict]:
    """
    Archivev10 Fix 4: Consolidate malformed multi-column checkbox fields.
    
    Detects fields with concatenated condition names in titles (e.g.,
    "Radiation Therapy Jaundice Jaw Joint Pain") and consolidates them
    into cleaner multi-select fields.
    """
    # Identify malformed fields in Medical/Dental History
    malformed_indices = []
    
    for i, item in enumerate(payload):
        section = item.get('section', '')
        title = item.get('title', '')
        item_type = item.get('type', '')
        
        # Only check dropdown fields in Medical/Dental History
        if section not in {'Medical History', 'Dental History'}:
            continue
        
        if item_type != 'dropdown':
            continue
        
        # Check if title looks malformed (3+ medical/dental terms concatenated)
        # Split title into words and count capitalized medical terms
        words = title.split()
        capitalized_words = [w for w in words if w and w[0].isupper() and len(w) > 2]
        
        # Check for medical/dental keywords
        medical_keywords = [
            'therapy', 'disease', 'disorder', 'condition', 'illness', 'syndrome',
            'pain', 'fever', 'bleeding', 'valve', 'joint', 'respiratory',
            'cardiovascular', 'hematologic', 'psychiatric', 'gastrointestinal',
            'arthritis', 'diabetes', 'asthma', 'seizure', 'allergy', 'nursing',
            'teeth', 'grinding', 'clenching', 'sucking', 'biting', 'chewing',
            'discolored', 'worn', 'crooked', 'spaces', 'overbite', 'sensitivity',
            'anesthesia', 'sulfa', 'drugs'
        ]
        
        title_lower = title.lower()
        keyword_count = sum(1 for kw in medical_keywords if kw in title_lower)
        
        # Enhanced detection: Malformed if title lacks connecting words
        # Good titles have "please mark", "do you", "have you", etc.
        has_instruction = bool(re.search(r'\b(please|mark|any|conditions|apply|do you|have you)\b', title_lower))
        
        # Malformed if: (4+ capitalized words AND no instructions) OR (4+ keywords AND no instructions)
        is_malformed = False
        if not has_instruction:
            if len(capitalized_words) >= 4 or keyword_count >= 4:
                is_malformed = True
        
        if is_malformed:
            # Also check that options exist and are reasonable
            options = item.get('control', {}).get('options', [])
            if len(options) >= 2:
                malformed_indices.append(i)
                if dbg:
                    dbg.gate(f"malformed_grid_detected -> '{title[:60]}...' with {len(options)} options")
    
    # If we found malformed fields, consolidate them by section
    if not malformed_indices:
        return payload
    
    # Group malformed fields by section
    by_section = {}
    for idx in malformed_indices:
        section = payload[idx].get('section', 'General')
        if section not in by_section:
            by_section[section] = []
        by_section[section].append(idx)
    
    # Consolidate each section's malformed fields
    to_remove = []
    new_fields = []
    
    for section, indices in by_section.items():
        # Skip if only 1 field (not worth consolidating)
        if len(indices) <= 1:
            continue
        
        # Collect all options from malformed fields
        all_options = []
        for idx in indices:
            options = payload[idx].get('control', {}).get('options', [])
            all_options.extend(options)
            to_remove.append(idx)
        
        # Remove duplicates
        seen = set()
        unique_options = []
        for opt in all_options:
            opt_name = opt.get('name', '') if isinstance(opt, dict) else opt
            if opt_name and opt_name.lower() not in seen:
                seen.add(opt_name.lower())
                unique_options.append(opt)
        
        # Create consolidated field
        if len(unique_options) >= 5:
            if section == "Medical History":
                title = "Medical History - Please mark any conditions that apply"
                key = "medical_conditions_consolidated"
            elif section == "Dental History":
                title = "Dental History - Please mark any conditions that apply"
                key = "dental_conditions_consolidated"
            else:
                title = f"{section} - Please mark any that apply"
                key = slugify(title)
            
            new_field = {
                'key': key,
                'title': title,
                'section': section,
                'type': 'dropdown',
                'optional': False,
                'control': {
                    'options': unique_options,
                    'multi': True
                }
            }
            
            new_fields.append((indices[0], new_field))  # Insert at position of first malformed field
            
            if dbg:
                dbg.gate(f"malformed_grid_consolidated -> {len(indices)} fields into '{title}' with {len(unique_options)} options")
    
    # Remove malformed fields in reverse order
    for idx in sorted(set(to_remove), reverse=True):
        payload.pop(idx)
    
    # Insert consolidated fields
    for insert_pos, new_field in sorted(new_fields, reverse=True):
        # Adjust insert position if we've already removed items
        adjusted_pos = insert_pos
        for removed_idx in sorted(to_remove):
            if removed_idx < insert_pos:
                adjusted_pos -= 1
        payload.insert(adjusted_pos, new_field)
    
    return payload


def postprocess_consolidate_continuation_options(payload: List[dict], dbg: Optional[DebugLogger] = None) -> List[dict]:
    """
    Archivev18 Fix 4: Consolidate checkbox fields that are continuations of previous fields.
    
    Pattern: Field with title that's just concatenated option names (e.g., "Local Anesthesia Sulfa Drugs Other")
    following a field with a proper question title in the same section.
    
    Example:
    Field 1: "Are you allergic to any of the following?" with options [Aspirin, Penicillin, ...]
    Field 2: "Local Anesthesia Sulfa Drugs Other" with options [Local Anesthesia, Sulfa Drugs, Other]
    
    Should consolidate Field 2's options into Field 1 and remove Field 2.
    """
    i = 0
    while i < len(payload):
        item = payload[i]
        title = item.get('title', '')
        
        # Check if this field looks like concatenated options (3+ capitalized words, no question marks/colons)
        # and is a dropdown with multiple options
        words = title.split()
        capitalized = [w for w in words if w and w[0].isupper()]
        
        is_concatenated = (
            len(capitalized) >= 3 and 
            '?' not in title and 
            not title.endswith(':') and
            item.get('type') in ('dropdown', 'radio')
        )
        
        if is_concatenated and i > 0:
            prev_item = payload[i - 1]
            prev_title = prev_item.get('title', '').lower()
            
            # Check if previous field is in same section and is a question about selecting/checking items
            same_section = item.get('section') == prev_item.get('section')
            is_selection_question = any(phrase in prev_title for phrase in [
                'allergic', 'any of the following', 'select', 'choose', 'check', 'mark'
            ])
            prev_is_dropdown = prev_item.get('type') in ('dropdown', 'radio')
            
            if same_section and is_selection_question and prev_is_dropdown:
                # Consolidate: add current field's options to previous field
                current_options = item.get('control', {}).get('options', [])
                prev_options = prev_item.get('control', {}).get('options', [])
                
                # Check if options in current field match parts of its title (confirmation it's concatenated)
                option_names = [opt.get('name', '') if isinstance(opt, dict) else opt for opt in current_options]
                title_has_options = sum(1 for opt_name in option_names if opt_name in title)
                
                if title_has_options >= 2:  # At least 2 option names appear in title
                    # Merge options
                    combined_options = prev_options + current_options
                    
                    # Remove duplicates
                    seen = set()
                    unique_options = []
                    for opt in combined_options:
                        opt_name = opt.get('name', '') if isinstance(opt, dict) else opt
                        if opt_name and opt_name.lower() not in seen:
                            seen.add(opt_name.lower())
                            unique_options.append(opt)
                    
                    prev_item['control']['options'] = unique_options
                    
                    if dbg:
                        dbg.gate(f"continuation_consolidated -> Merged '{title}' ({len(current_options)} opts) into '{prev_item['title']}' (total: {len(unique_options)} opts)")
                    
                    # Remove current field
                    payload.pop(i)
                    continue
        
        i += 1
    
    return payload


def postprocess_clean_overflow_titles(payload: List[dict], dbg: Optional[DebugLogger] = None) -> List[dict]:
    """
    Archivev11 Fix 3: Clean up field titles that have column overflow artifacts.
    
    Removes known label patterns that appear at the end of titles due to
    text extraction extending into adjacent columns.
    """
    LABEL_PATTERNS = [
        r'\s+Frequency\s*$',           # "Alcohol Frequency", "Drugs Frequency"
        r'\s+How\s+much\s*$',          # "How much"
        r'\s+How\s+long\s*$',          # "How long"
        r'\s+Comments?\s*:?\s*$',      # "Comments", "Comment:"
        r'\s+Additional\s+Comments?\s*:?\s*$',  # "Additional Comments"
        r'\s+Pattern\s*$',             # "Pattern"
        r'\s+Conditions?\s*$',         # "Conditions"
    ]
    
    for item in payload:
        title = item.get('title', '')
        original_title = title
        
        # Check if title ends with a known label pattern
        for pattern in LABEL_PATTERNS:
            if re.search(pattern, title, re.I):
                # Truncate at the pattern
                title = re.sub(pattern, '', title, flags=re.I).strip()
                
                if dbg and title != original_title:
                    dbg.gate(f"overflow_title_cleaned -> '{original_title}' → '{title}'")
                
                item['title'] = title
                break
    
    return payload


def postprocess_make_explain_fields_unique(payload: List[dict], dbg: Optional[DebugLogger] = None) -> List[dict]:
    """
    Archivev11 Fix 5: Make duplicate titles unique by adding context.
    
    When multiple fields have the same title in a section, add context from:
    - Preceding field (for explanation/follow-up fields)
    - Numeric suffix (for repeated fields like "Insured's Name")
    - Key information (as a last resort)
    """
    # Track title occurrences by section
    title_counts = {}
    
    for i, item in enumerate(payload):
        title = item.get('title', '').strip()
        section = item.get('section', 'General')
        key = item.get('key', '')
        
        # Create a unique identifier for this title in this section
        section_title = f"{section}:{title}"
        
        # Archivev18 Fix 3: Handle generic explanation titles even on first occurrence
        generic_titles = ['please explain', 'explanation', 'details', 'comments', 'if yes, please explain']
        is_generic = any(gt in title.lower() for gt in generic_titles)
        
        # If this is a generic title following a yes/no question, improve it immediately
        if is_generic and i > 0:
            prev_item = payload[i - 1]
            prev_title = prev_item.get('title', '')
            
            # If previous field is a yes/no question, use it as context
            if any(yn in prev_title.lower() for yn in ['yes or no', 'y or n', 'have you', 'are you', 'do you']):
                # Use full parent question title, but truncate if too long
                # Remove trailing question mark and colon for better readability
                context = prev_title.rstrip('?:').strip()
                
                # If context is too long (>60 chars), use first few words
                if len(context) > 60:
                    words = context.split()[:5]
                    context = ' '.join(words)
                
                new_title = f"{context} - Please explain"
                
                if dbg:
                    dbg.gate(f"unique_title -> '{title}' → '{new_title}' (context)")
                
                item['title'] = new_title
                # Update title for duplicate tracking
                title = new_title
                section_title = f"{section}:{title}"
        
        # Check if this is a duplicate
        if section_title in title_counts:
            count = title_counts[section_title]
            title_counts[section_title] = count + 1
            
            # Note: Generic titles are now handled above, before duplicate check
            
            # Strategy 2: For repeated fields (like Insurance fields), add numeric suffix
            # Check if key has a numeric suffix or scope marker
            key_has_suffix = bool(re.search(r'(_\d+|__\w+)$', key))
            if key_has_suffix:
                # Extract suffix info
                if '__primary' in key:
                    new_title = f"{title} (Primary)"
                elif '__secondary' in key:
                    new_title = f"{title} (Secondary)"
                elif re.search(r'_(\d+)$', key):
                    match = re.search(r'_(\d+)$', key)
                    num = match.group(1)
                    new_title = f"{title} #{num}"
                else:
                    new_title = f"{title} ({count + 1})"
                
                if dbg:
                    dbg.gate(f"unique_title -> '{title}' → '{new_title}' (suffix)")
                
                item['title'] = new_title
                continue
            
            # Strategy 3: Fallback - add numeric suffix
            new_title = f"{title} ({count + 1})"
            
            if dbg:
                dbg.gate(f"unique_title -> '{title}' → '{new_title}' (numeric)")
            
            item['title'] = new_title
        else:
            title_counts[section_title] = 1
    
    return payload

def postprocess_order_sections(payload: List[dict], dbg: Optional[DebugLogger] = None) -> List[dict]:
    """
    Order sections according to Modento conventions.
    
    Order: Patient Info → Insurance → Referral → Medical History → Consents → Signature
    
    This addresses the audit requirement that fields be organized in a logical,
    consistent order that matches user expectations and Modento best practices.
    """
    # Define the canonical section order
    SECTION_ORDER = {
        "Patient Information": 1,
        "Insurance": 2,
        "Referral": 3,
        "Medical History": 4,
        "Dental History": 5,
        "Consents": 6,
        "Signature": 7,
        # Default/catchall sections go at the end
        "General": 8,
    }
    
    def get_section_priority(field: dict) -> tuple:
        """
        Get sorting priority for a field.
        Returns tuple of (section_order, original_index) for stable sorting.
        """
        section = field.get("section", "General")
        # Get the priority from SECTION_ORDER, default to 99 for unknown sections
        priority = SECTION_ORDER.get(section, 99)
        return (priority, payload.index(field))
    
    # Sort the payload by section order
    sorted_payload = sorted(payload, key=get_section_priority)
    
    if dbg and dbg.enabled:
        # Log any reordering
        original_sections = [f.get("section", "General") for f in payload]
        sorted_sections = [f.get("section", "General") for f in sorted_payload]
        if original_sections != sorted_sections:
            dbg.gate(f"Section ordering: Reordered {len(payload)} fields")
            section_counts = {}
            for section in sorted_sections:
                section_counts[section] = section_counts.get(section, 0) + 1
            dbg.gate(f"  Section distribution: {section_counts}")
    
    return sorted_payload

def postprocess_validate_modento_compliance(payload: List[dict], dbg: Optional[DebugLogger] = None) -> List[dict]:
    """
    Final validation to ensure Modento schema compliance.
    
    Checks:
    1. All option values are non-empty
    2. Exactly one signature field with key "signature"
    3. Filter out footer/header/witness fields
    4. All keys are unique
    """
    issues = []
    seen_keys = set()
    signature_count = 0
    
    # Patterns for footer/header/witness fields that should be filtered out
    EXCLUDED_PATTERNS = [
        r'\bpractice\s+(name|phone|address|email)\b',
        r'\bwitness\b',
        r'\boffice\s+(phone|address|email|name)\b',
        r'\bfooter\b',
        r'\bheader\b',
        r'\bdoctor\s+(name|phone)\b',
        r'\bclinic\s+(name|phone|address)\b',
        r'\bfacility\s+(name|phone|address)\b',
    ]
    
    # First pass: filter out excluded fields and collect issues
    filtered_payload = []
    for idx, field in enumerate(payload):
        key = field.get("key", "")
        title = field.get("title", "").lower()
        field_type = field.get("type", "")
        
        # Check 3: Excluded fields (footer/header/witness) - FILTER THEM OUT
        is_excluded = False
        for pattern in EXCLUDED_PATTERNS:
            if re.search(pattern, title, re.I):
                issues.append(f"Filtered out excluded field: {title}")
                is_excluded = True
                break
        
        if is_excluded:
            continue  # Skip this field - don't add it to filtered_payload
        
        # Check 1: Duplicate keys
        if key in seen_keys:
            issues.append(f"Duplicate key: {key}")
        seen_keys.add(key)
        
        # Check 2: Signature uniqueness
        if field_type == "signature":
            signature_count += 1
            if key != "signature":
                issues.append(f"Signature field has incorrect key: {key} (should be 'signature')")
        
        # Check 4: Option values are non-empty
        if field_type in ["radio", "dropdown"]:
            options = field.get("control", {}).get("options", [])
            for opt_idx, opt in enumerate(options):
                value = opt.get("value")
                if value is None or value == "":
                    name = opt.get("name", f"option_{opt_idx}")
                    issues.append(f"Empty option value in {key}: option '{name}'")
        
        filtered_payload.append(field)
    
    # Check signature count
    if signature_count == 0:
        issues.append("No signature field found")
    elif signature_count > 1:
        issues.append(f"Multiple signature fields found: {signature_count} (should be exactly 1)")
    
    # Log issues if any
    if issues and dbg and dbg.enabled:
        dbg.gate(f"Modento compliance validation found {len(issues)} issues:")
        for issue in issues[:10]:  # Limit to first 10 issues
            dbg.gate(f"  - {issue}")
        if len(issues) > 10:
            dbg.gate(f"  ... and {len(issues) - 10} more")
    
    return filtered_payload


# ========== Parity Improvement Post-Processing Functions ==========

def postprocess_normalize_signatures(payload: List[dict]) -> List[dict]:
    """
    Parity Improvement #11: Ensure signature fields use consistent block_signature type.
    """
    for field in payload:
        field = normalize_signature_field(field)
    return payload


def postprocess_group_consent_fields(payload: List[dict], dbg: Optional[DebugLogger] = None) -> List[dict]:
    """
    Parity Improvement #9 & #10: Group consecutive consent/terms fields and risk lists.
    
    This consolidates:
    - Multiple "Terms", "Terms (2)", "Terms (3)" fields into fewer consent blocks
    - Risk/complication list items into single terms blocks
    """
    if not payload:
        return payload
    
    # First, group consecutive consent paragraphs
    grouped = group_consecutive_consent_paragraphs(payload)
    
    # Then, identify and group risk list sections
    final_payload = []
    i = 0
    while i < len(grouped):
        field = grouped[i]
        title = field.get('title', '')
        
        # Check if this starts a risk list
        if is_risk_list_header(title):
            # Group following risk items
            grouped, next_i = group_risk_list_items(grouped, i + 1)
            i = next_i
        else:
            final_payload.append(field)
            i += 1
    
    if dbg and dbg.enabled and len(final_payload) < len(payload):
        dbg.gate(f"consent_grouping -> Consolidated {len(payload)} fields to {len(final_payload)} fields")
    
    return final_payload


# ---------- Dictionary application + reporting

@dataclass
class Stats:
    total: int
    used: int
    counts_by_section: Dict[str,int]
    counts_by_type: Dict[str,int]
    matches: List[MatchEvent]
    near_misses: List[MatchEvent]

# ============================================================================
# SECTION 6: TEMPLATE APPLICATION AND I/O
# ============================================================================
# Functions for matching fields to templates and writing output files.

def apply_templates_and_count(payload: List[dict], catalog: Optional[TemplateCatalog], dbg: DebugLogger) -> Tuple[List[dict], int]:
    """
    Apply template matching and count matches.
    
    Enhanced to preserve conditional follow-up fields (Archivev8 Fix 3).
    """
    if not catalog:
        return payload, 0
    
    used = 0
    out: List[dict] = []
    
    for q in payload:
        # Archivev8 Fix 3: Check if this is a conditional/explanation field
        # Archivev15 Fix: Also skip opt-in preference fields (inline checkbox options)
        # PARITY FIX: Also skip witness signatures (they should not be matched against generic signature template)
        # These should not have templates applied to avoid breaking conditional relationships or changing keys
        is_conditional_field = (
            bool(q.get("conditional_on")) or
            "_explanation" in q.get("key", "") or
            "_followup" in q.get("key", "") or
            "_details" in q.get("key", "") or
            q.get("key", "").startswith("opt_in_") or  # Archivev15: Skip opt-in preference fields
            q.get("key", "") == "witness_signature" or  # PARITY FIX: Skip witness signatures
            (q.get("title", "").lower().strip() in ["please explain", "explanation", "details", "comments"])
        )
        
        if is_conditional_field:
            # Skip template matching for conditional fields
            out.append(q)
            if dbg.enabled and not q.get("key", "").startswith("opt_in_"):
                print(f"  [debug] template: skipping conditional field '{q.get('key')}' to preserve relationship")
            continue
        
        # Normal template matching for non-conditional fields
        fr = catalog.find(q.get("key"), q.get("title"), parsed_q=q)
        if fr.tpl:
            used += 1
            merged = merge_with_template(q, fr.tpl, scope_suffix=fr.scope)
            out.append(merged)
            dbg.log(MatchEvent(q.get("title",""), q.get("key",""), q.get("section",""), fr.tpl.get("key"), fr.reason, fr.score, fr.coverage))
        else:
            out.append(q)
            # Patch 4: Enhanced debug logging for unmatched fields (near-miss warnings)
            if fr.reason == "near":
                dbg.log_near(MatchEvent(q.get("title",""), q.get("key",""), q.get("section",""), fr.best_key, "near", fr.score, fr.coverage))
            elif dbg.enabled and q.get("title"):
                # Warn when fields are parsed but don't match any template
                # This helps identify missing dictionary entries
                print(f"  [warn] No dictionary match for field: '{q.get('title')}' (key: {q.get('key')})")
    
    out = _dedupe_keys_dicts(out)
    return out, used

def write_stats_sidecar(out_path: Path, payload: List[dict], used: int, dbg: DebugLogger, 
                        extraction_metadata: Optional[dict] = None, parsing_metadata: Optional[dict] = None):
    """
    Write enhanced statistics sidecar file.
    
    Priority 8.1: Enhanced Debug Output and Stats
    - Adds extraction metadata (file type, size, character count)
    - Adds parsing statistics (grids detected, options parsed)
    - Lists unmatched fields that might need dictionary entries
    """
    total = len(payload)
    counts_by_section = defaultdict(int)
    counts_by_type = defaultdict(int)
    for q in payload:
        counts_by_section[q.get("section","General")] += 1
        counts_by_type[q.get("type","input")] += 1
    
    # Priority 8.1: Collect unmatched fields for dictionary enhancement suggestions
    unmatched_fields = []
    if dbg.enabled:
        matched_keys = {ev.matched_key for ev in dbg.events if hasattr(ev, 'matched_key') and ev.matched_key}
        for q in payload:
            q_key = q.get("key", "")
            if q_key not in matched_keys:
                unmatched_fields.append({
                    "key": q_key,
                    "title": q.get("title", ""),
                    "section": q.get("section", ""),
                    "type": q.get("type", "")
                })
    
    stats = {
        "file": out_path.name,
        "total_items": total,
        "reused_from_dictionary": used,
        "reused_pct": (used/total*100.0 if total else 0.0),
        "counts_by_section": dict(counts_by_section),
        "counts_by_type": dict(counts_by_type),
        "matches": [ev.__dict__ for ev in dbg.events] if dbg.enabled else [],
        "near_misses": [ev.__dict__ for ev in dbg.near_misses] if dbg.enabled else [],
        "gates": dbg.gates if dbg.enabled else [],
    }
    
    # Priority 8.1: Add extraction metadata if available
    if extraction_metadata:
        stats["extraction"] = extraction_metadata
    
    # Priority 8.1: Add parsing metadata if available
    if parsing_metadata:
        stats["parsing"] = parsing_metadata
    
    # Priority 8.1: Add unmatched fields suggestions
    if unmatched_fields:
        stats["unmatched_fields"] = unmatched_fields
        stats["dictionary_suggestions"] = {
            "count": len(unmatched_fields),
            "message": "Consider adding these fields to dental_form_dictionary.json for better matching"
        }
    
    sidecar = out_path.with_suffix(".stats.json")
    sidecar.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------- IO

def process_one(txt_path: Path, out_dir: Path, catalog: Optional[TemplateCatalog] = None, debug: bool=False) -> Optional[Path]:
    raw = read_text_file(txt_path)
    if not raw.strip():
        print(f"[skip] empty file: {txt_path.name}")
        return None
    
    # Patch 3: Skip files that contain extraction error markers
    if raw.startswith("[NO TEXT LAYER]") or raw.startswith("[OCR NOT AVAILABLE]"):
        print(f"[skip] unextractable file: {txt_path.name} (no text layer and OCR unavailable)")
        return None
    
    # Priority 8.1: Collect extraction metadata
    extraction_metadata = {
        "source_file": txt_path.name,
        "file_size_bytes": txt_path.stat().st_size if txt_path.exists() else 0,
        "character_count": len(raw),
        "line_count": len(raw.split('\n'))
    }
    
    # Parse → normalize JSON
    questions = parse_to_questions(raw, debug=debug)
    
    # Priority 8.1: Collect parsing metadata
    parsing_metadata = {
        "raw_questions_parsed": len(questions),
        "sections_detected": len(set(q.section for q in questions)),
        "unique_sections": sorted(set(q.section for q in questions))
    }
    
    payload = questions_to_json(questions)

    # Post-process merges + normalization
    payload = postprocess_merge_hear_about_us(payload)
    payload = postprocess_consolidate_medical_conditions(payload)
    
    # Parity Improvement #11: Normalize signature fields
    payload = postprocess_normalize_signatures(payload)
    payload = postprocess_signature_uniqueness(payload)

    # Apply templates + count
    dbg = DebugLogger(enabled=debug)
    payload, used = apply_templates_and_count(payload, catalog, dbg)

    # Re-home after template merge
    payload = postprocess_rehome_by_key(payload, dbg=dbg)
    
    # New post-processing steps (Archivev9 fixes)
    payload = postprocess_infer_sections(payload, dbg=dbg)
    payload = postprocess_consolidate_duplicates(payload, dbg=dbg)
    
    # Parity Fix: Filter out document title fields that slipped through parsing
    payload = postprocess_filter_document_titles(payload, dbg=dbg)
    
    # Improvement #10: Enhanced duplicate consolidation
    payload = list(consolidate_duplicate_fields_enhanced(payload, debug=debug))
    
    # Improvement #11: Infer section boundaries
    payload = list(infer_section_boundaries(payload, debug=debug))
    
    # Archivev10 Fix 4: Consolidate malformed grid fields
    payload = postprocess_consolidate_malformed_grids(payload, dbg=dbg)
    
    # Archivev11 Fix 3: Clean up column overflow in field titles
    payload = postprocess_clean_overflow_titles(payload, dbg=dbg)
    
    # Archivev18 Fix 4: Consolidate continuation checkbox options
    payload = postprocess_consolidate_continuation_options(payload, dbg=dbg)
    
    # Archivev11 Fix 5: Make generic "Please explain" fields unique
    payload = postprocess_make_explain_fields_unique(payload, dbg=dbg)
    
    # Parity Improvement #9 & #10: Group consent and risk fields
    payload = postprocess_group_consent_fields(payload, dbg=dbg)
    
    # Modento Schema Compliance: Order sections according to conventions
    payload = postprocess_order_sections(payload, dbg=dbg)
    
    # Improvement #15: Add confidence scores to fields
    payload = add_confidence_scores(payload)
    
    # Performance Recommendation #2: Consolidate procedural consent blocks
    payload = consolidate_procedural_consent_blocks(payload)
    
    # Performance Recommendation #3: Enhance field type detection
    payload = [enhance_field_type_detection(field) for field in payload]
    
    # Modento Schema Compliance: Final validation
    payload = postprocess_validate_modento_compliance(payload, dbg=dbg)

    out_path = out_dir / (txt_path.stem + ".modento.json")
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    total = len(payload)
    pct = (used * 100 // total) if total else 0
    print(f"[✓] {txt_path.name} -> {out_path.name} ({total} items)")
    print(f"    ↳ reused {used}/{total} from dictionary ({pct}%)")

    if debug:
        dbg.print_summary()
    write_stats_sidecar(out_path, payload, used, dbg, extraction_metadata, parsing_metadata)
    return out_path

def process_one_wrapper(args_tuple):
    """
    Wrapper for process_one to support multiprocessing.
    
    Priority 6.1: Parallel Processing Support
    - Enables parallel processing of multiple forms
    - Returns tuple of (success, filename, error_message)
    
    Patch 4: Uses cached catalog for improved performance
    """
    txt_path, out_dir, dict_path, debug = args_tuple
    try:
        # Patch 4: Use cached catalog instead of loading each time
        catalog = None
        if dict_path and dict_path.exists():
            catalog = get_template_catalog(dict_path)
        
        process_one(txt_path, out_dir, catalog=catalog, debug=debug)
        return (True, txt_path.name, None)
    except Exception as e:
        return (False, txt_path.name, str(e))


# ============================================================================
# SECTION 7: MAIN ENTRY POINT
# ============================================================================
# Command-line interface and orchestration of the conversion pipeline.

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="in_dir",  default=DEFAULT_IN_DIR,  help="Folder with extracted .txt files (default: output)")
    ap.add_argument("--out", dest="out_dir", default=DEFAULT_OUT_DIR, help="Folder to write JSONs (default: JSONs)")
    ap.add_argument("--debug", action="store_true", help="Verbose debug logs + near-miss reporting; write *.stats.json sidecars")
    ap.add_argument("--jobs", type=int, default=1, help="Number of parallel jobs (default: 1 for sequential). Use -1 for CPU count.")
    args = ap.parse_args()

    in_dir = Path(args.in_dir); out_dir = Path(args.out_dir)
    if not in_dir.exists():
        print(f"ERROR: input folder not found: {in_dir}", file=sys.stderr); sys.exit(2)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load dictionary path (will be loaded in each worker if parallel)
    dict_path = Path(__file__).resolve().parent.parent / "dental_form_dictionary.json"
    
    txts = sorted([p for p in in_dir.rglob("*.txt") if p.is_file()])
    if not txts:
        print(f"WARNING: no .txt files found under {in_dir}")
        return
    
    # Priority 6.1: Determine number of jobs
    num_jobs = args.jobs
    if num_jobs == -1:
        import multiprocessing
        num_jobs = multiprocessing.cpu_count()
    
    # Sequential processing (default)
    if num_jobs <= 1:
        # Load dictionary once for sequential processing
        catalog: Optional[TemplateCatalog] = None
        if dict_path.exists():
            try:
                catalog = TemplateCatalog.from_path(dict_path)
            except Exception as e:
                print(f"[warn] dictionary unavailable ({e}). Proceeding without templates.")
        else:
            print(f"[warn] dictionary file not found at {dict_path.name}. Proceeding without templates.")
        
        for p in txts:
            try:
                process_one(p, out_dir, catalog=catalog, debug=args.debug)
            except Exception as e:
                print(f"[x] failed on {p.name}: {e}", file=sys.stderr)
    
    # Parallel processing (Priority 6.1)
    else:
        import multiprocessing
        print(f"Processing {len(txts)} file(s) with {num_jobs} parallel jobs...")
        
        # Prepare arguments for workers
        work_items = [(p, out_dir, dict_path, args.debug) for p in txts]
        
        # Process in parallel
        failed_files = []
        with multiprocessing.Pool(processes=num_jobs) as pool:
            results = pool.map(process_one_wrapper, work_items)
        
        # Report results
        successful = sum(1 for success, _, _ in results if success)
        for success, filename, error_msg in results:
            if not success:
                failed_files.append((filename, error_msg))
                print(f"[x] failed on {filename}: {error_msg}", file=sys.stderr)
        
        print(f"\n✅ Completed: {successful}/{len(txts)} files processed successfully")
        if failed_files:
            print(f"❌ Failed: {len(failed_files)} file(s) - see errors above")

if __name__ == "__main__":
    main()
