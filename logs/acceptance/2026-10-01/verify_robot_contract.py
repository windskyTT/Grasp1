from isaaclab.app import AppLauncher
launcher=AppLauncher(headless=True)
app=launcher.app
from pxr import Usd,UsdPhysics,UsdGeom
from pathlib import Path
import xml.etree.ElementTree as ET
import subprocess
asset=Path('/home/windsky/project/Grasp1/assets/robots/ur5_allegro')
old=ET.fromstring(subprocess.check_output(['git','show','HEAD:assets/robots/ur5_allegro/ur5_allegro.urdf']))
new=ET.parse(asset/'ur5_allegro.urdf').getroot()
for node in new:
    if node.tag in ('link','joint'):
        peer=next(p for p in old if p.tag==node.tag and p.attrib.get('name')==node.attrib['name'])
        node.tail=peer.tail=None
        assert ET.tostring(node)==ET.tostring(peer),(node.tag,node.attrib)
a=Usd.Stage.Open('/tmp/grasp1-table-baseline/assets/robots/ur5_allegro/ur5_allegro.usd')
b=Usd.Stage.Open(str(asset/'ur5_allegro.usd'))
for prim in Usd.PrimRange.Stage(b,Usd.TraverseInstanceProxies()):
    if prim.HasAPI(UsdPhysics.MassAPI):
        prev=a.GetPrimAtPath(prim.GetPath())
        for name in ('physics:mass','physics:centerOfMass','physics:diagonalInertia','physics:principalAxes'):
            assert prim.GetAttribute(name).Get()==prev.GetAttribute(name).Get(),(prim.GetPath(),name)
    if prim.IsA(UsdPhysics.RevoluteJoint):
        prev=a.GetPrimAtPath(prim.GetPath())
        for name in ('physics:axis','physics:lowerLimit','physics:upperLimit','physxJoint:maxJointVelocity'):
            assert prim.GetAttribute(name).Get()==prev.GetAttribute(name).Get(),(prim.GetPath(),name)
print('ROBOT_CONTRACT_PASS retained_link_joint_definitions_mass_inertia_limits',flush=True)
app.close()
