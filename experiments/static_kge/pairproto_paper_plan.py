"""Effective paper protocol with transparent, hash-checked seed extension."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def load_plan():
    base=HERE/'pairproto_paper_protocol.json'
    plan=json.loads(base.read_text(encoding='utf-8'))
    path=HERE/'pairproto_five_seed_amendment.json'
    if path.exists():
        amendment=json.loads(path.read_text(encoding='utf-8'))
        if hashlib.sha256(base.read_bytes()).hexdigest()!=amendment['base_protocol_sha256']:
            raise ValueError('Five-seed amendment does not match the frozen base protocol')
        plan['selection_seeds']=list(plan['ancillary_seeds'])
        plan['main_seeds']=list(amendment['evaluation_seeds'])
        plan['ancillary_seeds']=list(amendment['evaluation_seeds'])
        plan['five_seed_amendment']=amendment
        plan['selection']='All current-paper evaluation groups use five matched training seeds. Existing development selections are frozen; original search/calibration seed budgets are unchanged.'
    return plan
