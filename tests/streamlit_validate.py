def validate_input(hrs_studied,prev_score,extra_activities,sleep_hrs,sample_questions):
    errors = []
    if not (0 <= hrs_studied <=24):
        errors.append('Hrs Studied must be between 0 and 24')
    if not (0<=prev_score<=100):
        errors.append('Scores must be between 0 and 100')
    if extra_activities not in ['Yes','No']:
        errors.append('Extracurricular activities must be Yes or No')
    if not (0<=sleep_hrs<=24):
        errors.append('Sleep hours must be between 0 and 24')
    if not (sample_questions>=0):
        errors.append('Sample Questions must be a positive number')
    return errors



