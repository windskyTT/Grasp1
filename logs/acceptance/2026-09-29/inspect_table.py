from isaaclab.app import AppLauncher
app=AppLauncher(headless=True).app
from pxr import Usd
stage=Usd.Stage.Open('https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Props/Mounts/SeattleLabTable/table_instanceable.usd')
for p in Usd.PrimRange.Stage(stage,Usd.TraverseInstanceProxies()):
 print('TABLE_PRIM',p.GetPath(),p.GetTypeName(),p.GetAppliedSchemas(),flush=True)
app.close()
