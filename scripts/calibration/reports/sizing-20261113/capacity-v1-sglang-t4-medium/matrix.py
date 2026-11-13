import json,os,subprocess
from pathlib import Path
trials=json.loads(Path('/tmp/trials.json').read_text())
for i,trial in enumerate(trials):
    n=trial['tensor_parallel_size']
    env={**os.environ,'HFMEM_TP_SIZE':str(n),'HFMEM_GPU_INDICES':','.join(map(str,range(n))),'HFMEM_MAX_NUM_SEQS':str(trial['max_num_seqs']),'HFMEM_CASES_PATH':f'/tmp/cases-{i}.json','HFMEM_AUDIT_WEIGHTS':'1'}
    code=subprocess.call(['python3','-u','/tmp/driver.py'],env=env)
    print('HFMEM_RESULT '+json.dumps({'kind':'suite_exit','trial':i,'case':trial['case']['name'],'exit_code':code}),flush=True)
