"""Prediction CSV parsing and recorded six/three decimal serialization."""
import numpy as np


def parse(fields):
    tokens = fields.split()
    if tokens == ['none']: return []
    if len(tokens) % 6: raise ValueError('Malformed PredictionString')
    rows = []
    for i in range(0, len(tokens), 6):
        category = int(tokens[i]); score, x, y, w, h = map(float, tokens[i+1:i+6])
        if category not in range(10) or not 0 <= score <= 1 or min(w, h) <= 0:
            raise ValueError('Invalid class, score or box')
        if not np.isfinite([score, x, y, w, h]).all(): raise ValueError('Nonfinite box')
        rows.append({'category_id':category, 'score':score, 'bbox':[x,y,w,h]})
    return rows


def serialize(rows):
    fields=[]
    for r in sorted(rows,key=lambda r:r['score'],reverse=True):
        fields += [str(r['category_id']),f"{r['score']:.6f}"]
        fields += [f'{v:.3f}' for v in r['bbox']]
    return ' '.join(fields) if fields else 'none'
