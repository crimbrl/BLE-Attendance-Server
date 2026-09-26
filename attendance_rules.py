"""Pilot attendance transitions from ordered, five-second model predictions."""
ABSENT_WINDOWS_FOR_ALERT = 120  # 120 * 5 s = at least 10 minutes of valid ABSENT predictions.
PRESENT_STREAK = 3
STARTUP_WINDOWS = 6  # Initial 30 s is a calibration period, not an absence alert.
RULE_VERSION = "absence-120-present-3-v2"


def derive_events(rows, include_candidate=False):
    """Yield (window_index, event_type) for prolonged absence and return.

    Count ABSENT windows during a candidate departure. One or two PRESENT
    windows do not erase accumulated absence, but three consecutive PRESENT
    windows cancel the candidate or confirm a return. Interrupted/missing
    windows reset an unconfirmed candidate: they are not evidence of absence.
    """
    state = "UNKNOWN"
    previous_index = -1
    valid_streak = present_streak = absent_windows = 0
    candidate_index = None
    events = []

    for row in rows:
        index = row["window_index"]
        if index <= previous_index:
            raise ValueError("Window indexes must increase")
        if index != previous_index + 1:
            valid_streak = present_streak = absent_windows = 0
            candidate_index = None
        previous_index = index
        prediction = row["predicted_state"]
        if row["scan_state"] != "RUNNING" or prediction not in ("PRESENT", "ABSENT"):
            valid_streak = present_streak = absent_windows = 0
            candidate_index = None
            continue

        valid_streak += 1
        if prediction == "PRESENT":
            present_streak += 1
            if present_streak >= PRESENT_STREAK:
                absent_windows = 0
                candidate_index = None
        else:
            present_streak = 0
            if state == "PRESENT":
                if candidate_index is None:
                    candidate_index = index
                absent_windows += 1

        if state == "UNKNOWN":
            if valid_streak >= STARTUP_WINDOWS and present_streak >= PRESENT_STREAK:
                state = "PRESENT"
        elif state == "PRESENT" and absent_windows >= ABSENT_WINDOWS_FOR_ALERT:
            state = "ABSENT"
            absent_windows = 0
            candidate_index = None
            events.append((index, "LEFT"))
        elif state == "ABSENT" and present_streak >= PRESENT_STREAK:
            state = "PRESENT"
            absent_windows = 0
            candidate_index = None
            events.append((index, "RETURNED"))
    if include_candidate:
        return events, candidate_index
    return events
