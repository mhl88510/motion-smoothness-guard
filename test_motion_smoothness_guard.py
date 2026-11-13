import importlib.util,pathlib,unittest
P=pathlib.Path(__file__).with_name('motion_smoothness_guard.py');s=importlib.util.spec_from_file_location('guard',P);g=importlib.util.module_from_spec(s);s.loader.exec_module(g)
class TestDecision(unittest.TestCase):
 def base(self,**kw):
  d={'cfr_pass':True,'near_static_ratio':0.1,'max_consecutive_near_static_pairs':3,'max_consecutive_exact_duplicate_frames':0};d.update(kw);return d
 def test_pass_dynamic(self):self.assertEqual(g.decide(self.base())[0],'PASS')
 def test_return_ratio(self):self.assertIn('NEAR_STATIC_RATIO_EXCEEDED',g.decide(self.base(near_static_ratio=.41))[1])
 def test_return_long_run(self):self.assertIn('NEAR_STATIC_RUN_EXCEEDED',g.decide(self.base(max_consecutive_near_static_pairs=16))[1])
 def test_power_like_duplicate_cadence_passes(self):self.assertEqual(g.decide(self.base(near_static_ratio=.216,max_consecutive_near_static_pairs=7))[0],'PASS')
 def test_declared_static_parser(self):self.assertEqual(g.parse_ranges('[[1,2],[3,4.5]]'),[(1.0,2.0),(3.0,4.5)])
import json, shutil, subprocess, sys, tempfile
from unittest import mock
import numpy as np

class FrameMetricsTests(unittest.TestCase):
 def frames(self,values):return [np.full((2,2),v,dtype=np.uint8) for v in values]
 def measure(self,values,cuts=None,static=None):return g.measure_frames(iter(self.frames(values)),10,cuts or [],static or [],g.DEFAULTS)
 def test_dynamic_and_exact_pair_counts(self):
  m=self.measure([0,50,100,150]);self.assertEqual(m['near_static_pairs'],0);self.assertEqual(m['analyzed_pairs'],3)
  m=self.measure([0,0,0,0]);self.assertEqual(m['near_static_ratio'],1);self.assertEqual(m['max_consecutive_exact_duplicate_frames'],3)
 def test_exclusion_breaks_run(self):
  m=self.measure([0]*10,cuts=[.4]);self.assertEqual(m['excluded_pairs'],2);self.assertEqual(m['max_consecutive_near_static_pairs'],4)
 def test_intentional_static_resets_run(self):
  m=self.measure([0]*10,static=[(.3,.7)]);self.assertEqual(m['excluded_pairs'],4);self.assertEqual(m['max_consecutive_near_static_pairs'],3)
 def test_all_excluded_never_passes(self):
  m=self.measure([0]*5,static=[(0,2)]);m['cfr_pass']=True
  self.assertIn('NO_ANALYZABLE_FRAME_PAIRS',g.decide(m)[1])
 def test_single_frame_never_passes(self):
  m=self.measure([0]);m['cfr_pass']=True;self.assertEqual(g.decide(m)[0],'RETURN')
 def test_empty_input_has_no_pairs(self):self.assertEqual(self.measure([])['analyzed_pairs'],0)
 def test_invalid_ranges(self):
  for value in ['[[2,1]]','[[-1,2]]','[[0,NaN]]','[[0,Infinity]]','[[true,2]]','[[]]','{}']:
   with self.subTest(value=value),self.assertRaises(ValueError):g.parse_ranges(value)
 def test_invalid_cuts(self):
  for value in ['-1','[NaN]','[true]','[Infinity]']:
   with self.subTest(value=value),self.assertRaises(ValueError):g.parse_cuts(value)
 def test_cut_formats(self):
  self.assertEqual(g.parse_cuts('1,2,1'),[1.,2.]);self.assertEqual(g.parse_cuts('{"cuts":[1,2]}'),[1.,2.])
  with tempfile.TemporaryDirectory() as folder:
   p=pathlib.Path(folder)/'cuts.json';p.write_text('{"cut_times_seconds":[2,3]}');self.assertEqual(g.parse_cuts(str(p)),[2.,3.])
 def test_analyze_without_video(self):
  with mock.patch.object(g,'ffprobe',return_value={'streams':[]}):
   with self.assertRaisesRegex(ValueError,'no video'):g.analyze(pathlib.Path('unused'),[],[])
 def test_bad_frame_rate(self):
  probe={'streams':[{'codec_type':'video','r_frame_rate':'0/1','avg_frame_rate':'0/1'}],'format':{'duration':'1'}}
  with mock.patch.object(g,'ffprobe',return_value=probe):
   with self.assertRaises(ValueError):g.analyze(pathlib.Path('unused'),[],[])

@unittest.skipUnless(shutil.which('ffmpeg'),'FFmpeg is not on PATH')
class DecodeTests(unittest.TestCase):
 def test_real_synthetic_video_decode(self):
  with tempfile.TemporaryDirectory() as folder:
   video=pathlib.Path(folder)/'static.mkv'
   subprocess.run(['ffmpeg','-nostdin','-v','error','-f','lavfi','-i','color=c=black:s=32x32:r=10:d=1','-c:v','ffv1',str(video)],check=True)
   probe={'streams':[{'codec_type':'video','r_frame_rate':'10/1','avg_frame_rate':'10/1','nb_frames':'10'}],'format':{'duration':'1'}}
   with mock.patch.object(g,'ffprobe',return_value=probe):result=g.analyze(video,[],[])
   self.assertEqual(result['metrics']['frames'],10);self.assertEqual(result['status'],'RETURN')
   self.assertEqual(result['metrics']['exact_duplicate_pairs'],9)
 def test_decoder_error_is_reported(self):
  with tempfile.TemporaryDirectory() as folder:
   video=pathlib.Path(folder)/'invalid.mp4';video.write_bytes(b'not a video')
   with self.assertRaisesRegex(ValueError,'decoding failed'):list(g.decoded_frames(video,g.DEFAULTS))

@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'FFmpeg and ffprobe are required')
class IntegrationTests(unittest.TestCase):
 def test_full_command_line(self):
  with tempfile.TemporaryDirectory() as folder:
   root=pathlib.Path(folder);video=root/'static.mp4';report=root/'nested'/'report.json'
   subprocess.run(['ffmpeg','-nostdin','-v','error','-f','lavfi','-i','color=c=black:s=32x32:r=10:d=1','-c:v','libx264','-pix_fmt','yuv420p',str(video)],check=True)
   result=subprocess.run([sys.executable,str(P),'--video',str(video),'--out',str(report)],capture_output=True,text=True)
   self.assertEqual(result.returncode,2,result.stderr);self.assertEqual(json.loads(report.read_text())['status'],'RETURN')

if __name__=='__main__':unittest.main()
