"""只记录脚本物理基准窗口；不生成/加载policy checkpoint。"""
import ctypes,json,os,re,subprocess,time
from pathlib import Path
out=Path('logs/acceptance/optimization10/B/gui_scripted_visible_retry')
out.mkdir(parents=True,exist_ok=True)
command=['/home/windsky/IsaacLab/isaaclab.sh','-p','scripts/rsl_rl/scripted_grasp_probe.py','--mode','scripted','--num_envs','1','--trials','1','--seed','21','--output_dir',str(out)]
environment=dict(os.environ, DISPLAY=':1', HEADLESS='0')
captured=False
with (out/'gui.log').open('w') as log:
 process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=environment)
 while process.poll() is None:
  if not captured:
   states=[json.loads(line.split(' ',1)[1]) for line in (out/'gui.log').read_text().splitlines() if line.startswith('FULL_GRASP_STATE ')]
   if states and states[-1]['phase']==3 and states[-1]['hold_s']>.2:
    windows=subprocess.check_output(['xwininfo','-root','-tree'],env=environment,text=True)
    line=next(line for line in windows.splitlines() if 'Isaac Sim' in line)
    window=re.search(r'0x[0-9a-f]+',line).group()
    # 关闭已观测到挡住视野的Simulation Settings浮窗；不改变物理或控制。
    x11=ctypes.CDLL('libX11.so.6')
    xtst=ctypes.CDLL('libXtst.so.6')
    x11.XOpenDisplay.argtypes=[ctypes.c_char_p]
    x11.XOpenDisplay.restype=ctypes.c_void_p
    display=x11.XOpenDisplay(b':1')
    x11.XWarpPointer.argtypes=[ctypes.c_void_p,ctypes.c_ulong,ctypes.c_ulong,ctypes.c_int,ctypes.c_int,ctypes.c_uint,ctypes.c_uint,ctypes.c_int,ctypes.c_int]
    x11.XWarpPointer(display,0,int(window,16),0,0,0,0,416,160)
    xtst.XTestFakeButtonEvent.argtypes=[ctypes.c_void_p,ctypes.c_uint,ctypes.c_int,ctypes.c_ulong]
    xtst.XTestFakeButtonEvent(display,1,1,0)
    xtst.XTestFakeButtonEvent(display,1,0,0)
    x11.XFlush.argtypes=[ctypes.c_void_p]
    x11.XFlush(display)
    x11.XCloseDisplay.argtypes=[ctypes.c_void_p]
    x11.XCloseDisplay(display)
    time.sleep(.2)
    subprocess.run(['xwd','-silent','-id',window,'-out',str(out/'lift.xwd')],env=environment,check=True)
    subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(out/'lift.xwd'),'-frames:v','1',str(out/'lift.png')],check=True)
    (out/'capture_state.json').write_text(json.dumps(states[-1],indent=2))
    captured=True
  time.sleep(.2)
(out/'gui_capture_result.json').write_text(json.dumps(dict(returncode=process.returncode,captured=captured,
 controller='scripted',policy_attempts=0,checkpoint=None,command=command),indent=2))
print('SCRIPTED_GUI',process.returncode,captured)
