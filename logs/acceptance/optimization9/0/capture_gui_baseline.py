"""串行调用现有play.py并保存其窗口截图；历史Teacher只作为GUI接口基线。"""
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time

root = Path('logs/acceptance/optimization9/0')
folder = root/'gui_capture_checkpoint'
folder.mkdir()
original = Path('logs/rsl_rl/grasp1_ur5_allegro_teacher/2026-10-06_14-53-48_teacher_stability_action-clipping_none_stage1000/model_999.pt')
checkpoint = folder/original.name
shutil.copy2(original, checkpoint)
command = ['/home/windsky/IsaacLab/isaaclab.sh', '-p', 'scripts/rsl_rl/play.py',
           '--task', 'Grasp1-UR5-Allegro-Teacher-v0', '--checkpoint', str(checkpoint.resolve()),
           '--dataset', 'new_training_set', '--object', 'wood_block_oriented', '--num_envs', '1',
           '--episodes', '3', '--episode_length_s', '2', '--seed', '1', '--real-time',
           '--allow_mismatched_checkpoint', 'agent.clip_actions=null',
           'env.viewer.eye=[1.5,-2.0,1.6]', 'env.viewer.lookat=[0.2,-0.65,0.9]']
log = root/'gui_capture.log'
snapshot = root/'gui_baseline.png'
captured = False
ready_at = None
with log.open('w') as stdout:
    process = subprocess.Popen(command, stdout=stdout, stderr=subprocess.STDOUT)
    while process.poll() is None:
        if not captured and 'Critic MLP:' in log.read_text():
            ready_at = ready_at or time.monotonic()
            if time.monotonic() - ready_at > 1:
                windows = subprocess.check_output(['xwininfo', '-root', '-tree'], text=True)
                matches = [line for line in windows.splitlines() if 'Isaac Sim' in line]
                if matches:
                    window = re.search(r'0x[0-9a-f]+', matches[0]).group()
                    subprocess.run(['xwd', '-silent', '-id', window, '-out', str(root/'gui_baseline.xwd')], check=True)
                    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(root/'gui_baseline.xwd'),
                                    '-frames:v', '1', str(snapshot)], check=True)
                    (root/'gui_window.txt').write_text(matches[0] + '\n')
                    captured = True
        time.sleep(0.3)
result = dict(status='pass' if process.returncode == 0 and captured else 'fail', returncode=process.returncode,
              original_checkpoint=str(original.resolve()), diagnostic_copy=str(checkpoint.resolve()),
              command=shlex.join(command), screenshot=str(snapshot) if captured else None,
              full_grasp_gate='blocked', new_training_updates=0,
              purpose='历史Teacher纯策略推理的GUI接口基线，非本轮完整抓取成果')
(root/'gui_baseline_result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
with Path('logs/acceptance/optimization9/commands.md').open('a') as output:
    output.write('\n## GUI截图基线：历史Teacher，诊断语义覆盖\n```bash\n' + result['command'] + '\n```\n')
print(json.dumps(result, ensure_ascii=False), flush=True)
