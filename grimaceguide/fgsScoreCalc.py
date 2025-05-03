import numpy as np

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    if p1 is None or p2 is None: return None
    return np.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)

def calculate_angle(p1, p2, p3):
    """Calculate the angle at p2 formed by p1-p2-p3 in degrees."""
    if p1 is None or p2 is None or p3 is None: return None
    v1 = np.array([p1['x'] - p2['x'], p1['y'] - p2['y']])
    v2 = np.array([p3['x'] - p2['x'], p3['y'] - p2['y']])
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0: return None
    cos_theta = dot_product / (norm_v1 * norm_v2)
    cos_theta = np.clip(cos_theta, -1, 1)
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def calculate_vertical_angle(p1, p2):
    """Calculate the angle of the line p1-p2 relative to the vertical axis."""
    if p1 is None or p2 is None: return None
    delta_x = p2['x'] - p1['x']
    delta_y = p2['y'] - p1['y'] # Y increases downwards
    if delta_y == 0: # Horizontal line
        return 90.0
    angle_rad = np.arctan(delta_x / delta_y)
    angle_deg = np.degrees(angle_rad)
    # Adjust angle based on quadrant if needed, but for simple vertical comparison, this might suffice
    # We are interested in deviation from vertical (0 degrees)
    return abs(angle_deg)

def calculate_scores_from_landmarks(landmarks):
    """Calculate FGS scores based on labeled landmarks and FGS guidelines."""
    
    print("\n--- Calculating Scores (Using FGS Guidelines) ---")
    print(f"Received landmarks: {list(landmarks.keys())}")
    scores = {'ears': 0, 'eyes': 0, 'muzzle': 0, 'whiskers': 0, 'head': 0}
    face_width = None # Estimate face width for normalization

    # --- Pre-calculation: Estimate Face Width (e.g., between outer eye points) ---
    # Assuming left_eye_1 and right_eye_1 are outer points (NEEDS VERIFICATION)
    le1 = landmarks.get('left_eye_1')
    re1 = landmarks.get('right_eye_1')
    if le1 and re1:
        face_width = calculate_distance(le1, re1)
        print(f"Debug - Estimated Face Width: {face_width}")
    else:
        print("Warning: Could not estimate face width (missing eye_1 points).")


    # --- Ear Score ---
    # 0: Forward; 1: Slightly apart; 2: Flattened/Rotated outwards
    print("\n--- Ear Score Calculation ---")
    le_base = landmarks.get('left_ear_1')
    le_tip = landmarks.get('left_ear_3')
    le_outer = landmarks.get('left_ear_5')
    re_base = landmarks.get('right_ear_1')
    re_tip = landmarks.get('right_ear_3')
    re_outer = landmarks.get('right_ear_5')

    if le_base and le_tip and le_outer and re_base and re_tip and re_outer:
        # Angle relative to vertical (0=straight up, 90=horizontal)
        left_ear_angle = calculate_vertical_angle(le_base, le_tip)
        right_ear_angle = calculate_vertical_angle(re_base, re_tip)
        avg_ear_angle = (left_ear_angle + right_ear_angle) / 2 if left_ear_angle is not None and right_ear_angle is not None else None
        print(f"Debug - Avg Ear Angle from Vertical: {avg_ear_angle}")

        # Horizontal distance between tips
        tip_dist = calculate_distance(le_tip, re_tip)
        base_dist = calculate_distance(le_base, re_base)
        tip_base_ratio = tip_dist / base_dist if base_dist and base_dist > 0 else None
        print(f"Debug - Ear Tip/Base Distance Ratio: {tip_base_ratio}")

        # Scoring Logic (Example - needs tuning)
        if avg_ear_angle is not None and tip_base_ratio is not None:
            if avg_ear_angle > 45 or tip_base_ratio > 1.5: # Flattened / wide apart
                 scores['ears'] = 2
            elif avg_ear_angle > 25 or tip_base_ratio > 1.2: # Slightly apart
                 scores['ears'] = 1
            else: # Assume forward
                 scores['ears'] = 0
        else:
            print("Warning: Could not calculate ear metrics.")
            scores['ears'] = 0 # Default
    else:
        print("Warning: Missing some ear landmarks for scoring.")
        scores['ears'] = 0
    print(f"Debug - Ear Score: {scores['ears']}")


    # --- Eye Score ---
    # 0: Open; 1: Partially open; 2: Squinted
    print("\n--- Eye Score Calculation ---")
    # Assuming 1,2 are top lid; 3,4 are bottom lid (NEEDS VERIFICATION)
    le1, le2, le3, le4 = [landmarks.get(f'left_eye_{i+1}') for i in range(4)]
    re1, re2, re3, re4 = [landmarks.get(f'right_eye_{i+1}') for i in range(4)]

    if all([le1, le2, le3, le4, re1, re2, re3, re4]):
        # Calculate vertical opening and horizontal width
        left_top_mid_y = (le1['y'] + le2['y']) / 2
        left_bot_mid_y = (le3['y'] + le4['y']) / 2
        left_v_dist = abs(left_bot_mid_y - left_top_mid_y)
        left_h_dist = calculate_distance(le1, le2) # Or use le1 to le4? Depends on points

        right_top_mid_y = (re1['y'] + re2['y']) / 2
        right_bot_mid_y = (re3['y'] + re4['y']) / 2
        right_v_dist = abs(right_bot_mid_y - right_top_mid_y)
        right_h_dist = calculate_distance(re1, re2) # Or use re1 to re4?

        avg_v_dist = (left_v_dist + right_v_dist) / 2
        avg_h_dist = (left_h_dist + right_h_dist) / 2 if left_h_dist and right_h_dist else None

        print(f"Debug - Avg Eye Vertical Dist: {avg_v_dist}")
        print(f"Debug - Avg Eye Horizontal Dist: {avg_h_dist}")

        if avg_h_dist and avg_h_dist > 0:
            eye_aspect_ratio = avg_v_dist / avg_h_dist
            print(f"Debug - Eye Aspect Ratio (V/H): {eye_aspect_ratio}")

            # Scoring Logic (Example - needs tuning)
            if eye_aspect_ratio < 0.25: # Very small vertical opening relative to width
                scores['eyes'] = 2
            elif eye_aspect_ratio < 0.5: # Moderately open
                scores['eyes'] = 1
            else: # Wide open
                scores['eyes'] = 0
        else:
            print("Warning: Could not calculate eye aspect ratio.")
            scores['eyes'] = 0 # Default
    else:
        print("Warning: Missing some eye landmarks for scoring.")
        scores['eyes'] = 0
    print(f"Debug - Eye Score: {scores['eyes']}")


    # --- Muzzle Score ---
    # 0: Relaxed (round); 1: Mildly tense; 2: Tense (elliptical)
    print("\n--- Muzzle Score Calculation ---")
    nose_tip = landmarks.get('nose_3') # Assuming nose_3 is the tip
    mouth_left = landmarks.get('mouth_1') # Assuming mouth_1 is left corner
    mouth_right = landmarks.get('mouth_4') # Assuming mouth_4 is right corner
    # Could also use mouth_2, mouth_3, mouth_5, mouth_6 for curvature/shape

    if nose_tip and mouth_left and mouth_right:
        mouth_width = calculate_distance(mouth_left, mouth_right)
        mouth_center_x = (mouth_left['x'] + mouth_right['x']) / 2
        mouth_center_y = (mouth_left['y'] + mouth_right['y']) / 2
        mouth_center = {'x': mouth_center_x, 'y': mouth_center_y}
        nose_to_mouth_dist = calculate_distance(nose_tip, mouth_center) # Vertical distance proxy

        print(f"Debug - Mouth Width: {mouth_width}")
        print(f"Debug - Nose to Mouth Center Distance: {nose_to_mouth_dist}")

        if mouth_width and nose_to_mouth_dist and nose_to_mouth_dist > 0:
             # Ratio: Wider mouth relative to vertical distance might indicate tension/flattening
             muzzle_ratio = mouth_width / nose_to_mouth_dist
             print(f"Debug - Muzzle Ratio (Width/NoseDist): {muzzle_ratio}")

             # Scoring Logic (Example - needs tuning)
             if muzzle_ratio > 1.5: # Mouth significantly wider than nose-mouth distance
                 scores['muzzle'] = 2
             elif muzzle_ratio > 1.1: # Mouth somewhat wider
                 scores['muzzle'] = 1
             else: # Assume relaxed (roundish)
                 scores['muzzle'] = 0
        else:
             print("Warning: Could not calculate muzzle ratio.")
             scores['muzzle'] = 0 # Default
    else:
        print("Warning: Missing some muzzle landmarks for scoring.")
        scores['muzzle'] = 0
    print(f"Debug - Muzzle Score: {scores['muzzle']}")


    # --- Whiskers Score ---
    # 0: Loose/curved; 1: Slightly curved/straight; 2: Straight/Forward
    print("\n--- Whiskers Score Calculation ---")
    lw1, lw3, lw5 = landmarks.get('left_whisker_1'), landmarks.get('left_whisker_3'), landmarks.get('left_whisker_5')
    rw1, rw3, rw5 = landmarks.get('right_whisker_1'), landmarks.get('right_whisker_3'), landmarks.get('right_whisker_5')
    nose_base_l = landmarks.get('nose_1') # Proxy for whisker pad area start?
    nose_base_r = landmarks.get('nose_5') # Proxy for whisker pad area start?

    if all([lw1, lw3, lw5, rw1, rw3, rw5, nose_base_l, nose_base_r]):
        # Horizontal Spread
        left_spread = abs(lw5['x'] - lw1['x'])
        right_spread = abs(rw5['x'] - rw1['x'])
        avg_spread = (left_spread + right_spread) / 2
        print(f"Debug - Avg Whisker Spread: {avg_spread}")

        # Forward position (compare X of tips lw5/rw5 to X of base lw1/rw1 or nose points)
        left_forward = lw5['x'] < lw1['x'] # Is tip further left (forward on left side)?
        right_forward = rw5['x'] > rw1['x'] # Is tip further right (forward on right side)?
        print(f"Debug - Whiskers Forward (L/R): {left_forward} / {right_forward}")

        # Curvature (simplified): Check if midpoint Y is lower (droopier) than endpoints Y
        left_curve_proxy = lw3['y'] > (lw1['y'] + lw5['y']) / 2
        right_curve_proxy = rw3['y'] > (rw1['y'] + rw5['y']) / 2
        print(f"Debug - Whiskers Curved Proxy (L/R): {left_curve_proxy} / {right_curve_proxy}")

        # Normalize spread? Use face_width if available
        normalized_spread = avg_spread / face_width if face_width else None
        print(f"Debug - Normalized Whisker Spread: {normalized_spread}")

        # Scoring Logic (Example - needs tuning)
        # Score 2: Straight (not curved) AND Forward
        # Score 0: Curved AND not particularly forward (or wide spread?)
        # Score 1: In between
        is_straight = not (left_curve_proxy or right_curve_proxy)
        is_forward = left_forward and right_forward

        if is_straight and is_forward:
            scores['whiskers'] = 2
        elif not is_straight and not is_forward: # Curved and not forward
             # Check spread? Maybe wide spread = relaxed?
             if normalized_spread and normalized_spread > 0.8: # Wide spread relative to face
                 scores['whiskers'] = 0
             else: # Curved but maybe not wide?
                 scores['whiskers'] = 1 # Uncertainty
        else: # Mix of straight/curved/forward
            scores['whiskers'] = 1 # Default to moderate/uncertain

    else:
        print("Warning: Missing some whisker landmarks for scoring.")
        scores['whiskers'] = 0 # Default
    print(f"Debug - Whiskers Score: {scores['whiskers']}")


    # --- Head Score ---
    # 0: Above shoulder; 1: Aligned; 2: Below shoulder/Tilted
    print("\n--- Head Score Calculation ---")
    chin_point = landmarks.get('chin_point')
    left_ear_base = landmarks.get('left_ear_1') # Proxy for shoulder line
    right_ear_base = landmarks.get('right_ear_1') # Proxy for shoulder line

    if chin_point and left_ear_base and right_ear_base:
        avg_ear_base_y = (left_ear_base['y'] + right_ear_base['y']) / 2
        chin_y = chin_point['y']
        print(f"Debug - Chin Y: {chin_y}")
        print(f"Debug - Avg Ear Base Y (Shoulder Proxy): {avg_ear_base_y}")

        # Calculate vertical distance for normalization (e.g., ear base to chin)
        head_height_proxy = abs(chin_y - avg_ear_base_y)
        print(f"Debug - Head Height Proxy: {head_height_proxy}")

        # Scoring Logic (Example - needs tuning)
        # Compare chin_y relative to avg_ear_base_y
        # Higher Y means lower on the image
        diff = chin_y - avg_ear_base_y
        normalized_diff = diff / head_height_proxy if head_height_proxy and head_height_proxy > 0 else 0
        print(f"Debug - Normalized Head Position (Chin relative to Ear Base): {normalized_diff}")

        # Thresholds based on normalized difference (relative position)
        if normalized_diff > 0.1: # Chin significantly lower than ear base line
            scores['head'] = 2
        elif normalized_diff > -0.1: # Chin roughly aligned with ear base line
            scores['head'] = 1
        else: # Chin significantly higher than ear base line
            scores['head'] = 0
    else:
        print("Warning: Missing landmarks for head scoring.")
        scores['head'] = 0
    print(f"Debug - Head Score: {scores['head']}")

    # Calculate total score
    total_score = sum(scores.values())
    scores['total_score'] = total_score

    print(f"\nCalculated Scores: {scores}")
    print("--- End Score Calculation ---")
    return scores
