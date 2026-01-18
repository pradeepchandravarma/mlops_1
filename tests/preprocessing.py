def encode_extra_activities(extra_activities):
    mapping = {'Yes':1,'No':0}
    if extra_activities not in mapping:
        raise ValueError('Extracurricular Activities must be Yes or No')
    return mapping[extra_activities]

    
