import pandas as pd
import numpy as np
import joblib
import os
from openai import OpenAI
from fuzzywuzzy import process
from dotenv import load_dotenv
import re

model_path = os.path.join('models/debt_model.pkl')

debt_model = joblib.load(model_path)


#API KEY
load_dotenv()  # load environment variables from .env

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

earnings_df = pd.read_csv("data/Most-Recent-Cohorts-Field-of-Study.csv", low_memory=False)
df_inst = pd.read_csv("data/Most-Recent-Cohorts-Institution.csv", low_memory = False)

earnings_df = earnings_df[
    (earnings_df['EARN_MDN_5YR'].notna()) &
    (earnings_df['EARN_MDN_5YR'] != 'PrivacySuppressed') &
    (earnings_df['EARN_MDN_5YR'] != 'NULL') &
    (earnings_df['EARN_MDN_5YR'] != 'PS')
]
earnings_df.loc[:,'EARN_MDN_5YR'] = earnings_df['EARN_MDN_5YR'].astype(float)



#All the helper functions
def fuzzy_match_college_name(user_input: str, college_list: list, threshold: int = 80):
    """
    Matches user input to closest college name in your list using fuzzy string matching.
    
    Args:
        user_input (str): what the user typed (e.g., 'uci', 'berkly')
        college_list (list): list of official names from your dataset
        threshold (int): minimum score to accept the match

    Returns:
        str or None: best matched college name, or None if not found
    """
    match, score = process.extractOne(user_input.lower(), [name.lower() for name in college_list])
    #print(f'match: {match}, score: {score}')
    return match if score >= threshold else None

def get_model_input_for_debt(college_name: str, income_tier: int, df_institution: pd.DataFrame):
    # Match user input to official college name (fuzzy or direct)
    match = fuzzy_match_college_name(college_name, df_institution['INSTNM'].dropna().unique().tolist())
    if not match:
        raise ValueError("Could not find a matching college.")

    # Get the institution row
    row = df_institution[df_institution['INSTNM'].str.lower() == match.lower()].iloc[0]

    features = {
        'COSTT4_A': row['COSTT4_A'],
        'TUITIONFEE_IN': row['TUITIONFEE_IN'],
        'TUITIONFEE_OUT': row['TUITIONFEE_OUT'],
        'ROOMBOARD_ON': row['ROOMBOARD_ON'],
        'BOOKSUPPLY': row['BOOKSUPPLY'],
        'CONTROL': row['CONTROL'],
        'SAT_AVG': row['SAT_AVG'],
        'ADM_RATE': row['ADM_RATE'],
        'UGDS': row['UGDS'],
        'LOCALE': row['LOCALE'],
        'PCTPELL': row['PCTPELL'],
        'PCTFLOAN': row['PCTFLOAN'],
        'FTFTPCTPELL': row['FTFTPCTPELL'],
        'REGION': row['REGION'],
        'C150_4': row['C150_4'],
        'income_tier': income_tier
    }

    return features  # This goes into your predict_raw_debt()

MODEL_FEATURES = [
    'COSTT4_A',
    'TUITIONFEE_IN',
    'TUITIONFEE_OUT',
    'ROOMBOARD_ON',
    'BOOKSUPPLY',
    'CONTROL',
    'SAT_AVG',
    'ADM_RATE',
    'UGDS',
    'LOCALE',
    'PCTPELL',
    'PCTFLOAN',
    'FTFTPCTPELL',
    'REGION',
    'C150_4',
    'income_tier'
]
def predict_raw_debt(user_features: dict):
    """
    Takes user features (including income_tier) and returns the model's predicted debt.
    
    Args:
        user_features (dict): dictionary of all required model features
    
    Returns:
        float: predicted median student debt (unadjusted)
    """

    # Safety check to make sure all features are there
    missing = [f for f in MODEL_FEATURES if f not in user_features]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    # Convert to DataFrame in the correct order
    input_df = pd.DataFrame([[user_features[f] for f in MODEL_FEATURES]], columns=MODEL_FEATURES)

    # Predict using your model
    prediction = debt_model.predict(input_df)[0]

    return round(prediction, 2)



def adjust_debt(predicted_debt, in_state_tuition, tuition_out, control, is_user_in_state, parent_loans):
    parent_loan_amt = 29000
    if control == 1:
        avg_tuition = 4 * (0.7 * in_state_tuition + 0.3 * tuition_out)
        user_tuition = 4 * (in_state_tuition if is_user_in_state else tuition_out)
        tuition_ratio = user_tuition / avg_tuition

        adjusted_debt = predicted_debt*tuition_ratio
        if parent_loans:
            adjusted_debt += (parent_loan_amt*0.75)
        return adjusted_debt
    else:
        if parent_loans:
            return predicted_debt + parent_loan_amt
        return predicted_debt


def get_earnings(college_name, major):
    row = earnings_df[
        (earnings_df['INSTNM'].str.lower() == college_name.lower()) &
        (earnings_df['CIPDESC'].str.lower() == major.lower())
    ]
    return float(row['EARN_MDN_5YR'].values[0]) if not row.empty else None


import re

def get_qol_score(college_name):
    prompt = f"""
    Based on data that you get from searching Niche.com, student reviews, and several Google websites, assign a Quality of Life score (1 to 10) for {college_name}.
    Include a short reason in 1–2 sentences.
    Respond as: QoL Score: X/10. Explanation: ...
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that evaluates college quality of life."},
            {"role": "user", "content": prompt}
        ]
    )

    message = response.choices[0].message.content.strip()

    # Regex pattern to extract "QoL Score: X/10. Explanation: ..."
    match = re.match(r"QoL Score:\s*(\d+)/10\.?\s*Explanation:\s*(.*)", message, re.IGNORECASE)

    if match:
        score = int(match.group(1))
        explanation = match.group(2).strip()
        return score, explanation
    else:
        # Fallback if format doesn't match
        return None, message




def compute_roi(earn_mdn_5yr, adjusted_debt):
  if adjusted_debt <= 0:
        return 0  # avoid division by 0 or weird negative debt
  total_earnings = earn_mdn_5yr * 5
  return total_earnings / adjusted_debt



def normalize_roi(roi_raw, cap=10):
    if roi_raw <= 0:
        return 1
    score = np.log1p(roi_raw) * 2.0
    return min(round(score, 2), cap)



def fuzzy_match(value, choices, threshold=80):
    """
    Generic fuzzy matcher to find the closest match from a list.
    Returns None if no match is above threshold.
    """
    match, score = process.extractOne(value.lower(), [c.lower() for c in choices])
    return match if score >= threshold else None

def get_earnings_fuzzy_custom(user_college, user_major, threshold=80):
    """
    Uses fuzzy string matching on both college and major names to get EARN_MDN_5YR.
    """
    # Get fuzzy match for college name
    matched_college = fuzzy_match_college_name(user_college, df_inst['INSTNM'].dropna().unique(), threshold)
    if not matched_college:
        print("No close match for college.")
        return None

    df_college = earnings_df[earnings_df['INSTNM'].str.lower().str.strip() == matched_college.lower()]

    # Get fuzzy match for major name
    matched_major = fuzzy_match(user_major, df_college['CIPDESC'].dropna().unique(), threshold)
    if not matched_major:
        print("No close match for major.")
        return None

    df_result = df_college[df_college['CIPDESC'].str.lower().str.strip() == matched_major.lower()]
    #print(f"Matched college: {matched_college}")
    #print(f"Matched major: {matched_major}")

    if not df_result.empty:
        try:
            return float(df_result['EARN_MDN_5YR'].values[0])
        except:
            print("EARN_MDN_5YR could not be converted.")
            return None

    print("Final matched row not found.")
    return None



def is_in_state(user_state, college, df = df_inst):
    match = fuzzy_match_college_name(college, df['INSTNM'].dropna().unique())
    if not match:
        raise ValueError("Could not find a matching college.")
    
    college_row = df[df['INSTNM'].str.lower() == match.lower()]
    if college_row.empty:
        raise ValueError("Matching college not found in institution dataset.")

    college_state = college_row.iloc[0]['STABBR'].strip().upper()
    return college_state == user_state.strip().upper()

def evaluate_college(
    college_list,
    major,
    income_tier,
    user_state,
    parent_loans,
    weight_qol,
    weight_roi
):
    final_scores = {}
    details = {}  # Save scores per college for explanation

    for college_name in college_list:
        try:
            # 1. Predict debt
            is_user_in_state = is_in_state(user_state, college_name)
            user_features = get_model_input_for_debt(college_name, income_tier, df_institution=df_inst)
            predicted_debt = predict_raw_debt(user_features)
            in_state_tuition = user_features['TUITIONFEE_IN']
            tuition_out = user_features['TUITIONFEE_OUT']
            control = user_features['CONTROL']
            #print(f"[{college_name}] predicted_debt: {predicted_debt}")
            #print(f"[{college_name}] tuition_in: {in_state_tuition}, tuition_out: {tuition_out}, control: {control}")

            adjusted_debt = adjust_debt(predicted_debt, in_state_tuition, tuition_out, control, is_user_in_state, parent_loans)

            # 2. Earnings & ROI
            earn = get_earnings_fuzzy_custom(college_name, major)
            raw_roi = compute_roi(earn, adjusted_debt)
            normalized_roi = normalize_roi(raw_roi)

            # 3. QoL
            qol_score, qol_explanation = get_qol_score(college_name)

            # 4. Weighted score
            final_score = weight_roi * normalized_roi + weight_qol * qol_score
            final_scores[college_name] = final_score

            # Save for explanation
            details[college_name] = {
                "roi_score": normalized_roi,
                "qol_score": qol_score,
                "adjusted_debt": adjusted_debt,
                "earnings": earn,
                "qol_explanation": qol_explanation,
                "final_score": round(final_score, 2)
            }

        except Exception as e:
            print(f"Error processing {college_name}: {e}")
            continue

    if not final_scores:
        return {"error": "No valid college scores calculated."}

    # 5. Get best college
    best_name = max(final_scores, key=final_scores.get)
    best = details[best_name]

    # 6. Generate OpenAI explanation
    prompt = f"""
      A student is deciding between the following colleges to major in {major}.
      They value both Return on Investment (ROI) and Quality of Life (QoL), with a preference weighting of {int(weight_qol*100)}% QoL and {int(weight_roi*100)}% ROI.

      Each college below was evaluated based on:
      - **Predicted student debt**
      - **Expected median earnings 5 years after graduation**
      - **Quality of Life**, based on student reviews and public sentiment.

      Here are the scores:
      {"".join([f"{name}: ROI Score = {round(d['roi_score'], 2)}/10, QoL Score = {d['qol_score']}/10" for name, d in details.items()])}

      The best overall recommendation is **{best_name}**, with an ROI of {round(best['roi_score'], 2)}/10 and QoL of {best['qol_score']}/10.

      Write a concise, insightful explanation (3-5 sentences) for why this college was chosen. Be sure to explain *why* the ROI and QoL scores are high or low — including factors like tuition cost, predicted debt, expected earnings, and quality-of-life indicators (campus, safety, satisfaction, etc.).
      """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful college advisor."},
            {"role": "user", "content": prompt}
        ]
    )

    explanation = response.choices[0].message.content.strip()

    return {
        "best_college": best_name,
        "final_score": float(round(final_scores[best_name], 2)),
        "roi_score": float(best['roi_score']),
        "qol_score": int(best['qol_score']),
        "adjusted_debt": float(best['adjusted_debt']),
        "median_earnings": float(best['earnings']),
        "explanation": explanation
    }


