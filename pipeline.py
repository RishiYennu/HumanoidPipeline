import sys
import time
from google import genai
from google.genai import types
from PIL import Image
import subprocess
import os

HOME = os.path.expanduser("~")
GVHMR_DIR = os.path.join(HOME, "GVHMR")
GMR_DIR = os.path.join(HOME, "GMR")

# Create a file within the folder gemini_files and name is apiKey and paste only your Gemini API key in there.
f = open("gemini_files/apiKey", "r")
client = genai.Client(api_key=f.read())

def run(cmd, cwd=None, env=None):
    """Run a shell command and stream output."""
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, env=env)
    if result.returncode != 0:
        print(f"WARNING: Command exited with code {result.returncode}")
    return result.returncode


def main():
  args = sys.argv[1:]

  if len(args) > 1 or len(args) == 0:
    print("The command for running the script is: python pipeline.py [simulation image]")
    return
  
  image = sys.argv[1]

  human_image = generate_human_image(image)

  generate_video(human_image)

  motion_retargeting("output/output.mp4")


def generate_human_image(robot_image : str):
  # Edit Image prompt in the below file
  image_prompt = open("gemini_files/imagePrompt.txt", "r")
  img = Image.open(robot_image)

  response = client.models.generate_content(
      model="gemini-2.0-flash-exp-image-generation",
      contents=[image_prompt.read(), img],
      config=types.GenerateContentConfig(
          response_modalities=["TEXT", "IMAGE"],
      ),
  )

  for part in response.candidates[0].content.parts:
      if part.inline_data:
          with open("output/human_video_image.png", "wb") as f:
              f.write(part.inline_data.data)
          print("Saved human_video_image.png")
      elif part.text:
          print(part.text)
  
  return "output/human_video_image.png"

def generate_video(human_image : str): 
  with open(human_image, "rb") as f:
      image_bytes = f.read()
  
  image = types.Image(
      image_bytes=image_bytes,
      mime_type="image/png",
  )
  
  operation = client.models.generate_videos(
      model="veo-3.0-generate-001",
      image=image,
      prompt="Describe the motion/animation you want",
      config=types.GenerateVideosConfig(
          number_of_videos=1,
          duration_seconds=8, 
          negative_prompt="blurry, low quality",  
      ),
  )
  
  # Try both response and result
  while not operation.done:
      time.sleep(10)
      operation = client.operations.get(operation)
  
  # Try both response and result
  generated = None
  if operation.response and operation.response.generated_videos:
      generated = operation.response.generated_videos
  elif operation.result and operation.result.generated_videos:
      generated = operation.result.generated_videos
  
  if not generated:
      print("Video generation failed - no videos returned")
      print(f"Operation: {operation}")
      return
  
  client.files.download(file=generated[0].video)
  generated[0].video.save("output/output.mp4")
  print("Saved output.mp4")
  

def motion_retargeting(input_video_dir):
  if not (os.path.isdir(GVHMR_DIR) and os.path.isdir(GMR_DIR)):
    print("Please run the [install.py] file to install GVHMR and GMR in order to perform the retargeting")
    return
  
  # Convert to absolute path so it works from any cwd
  input_video_dir = os.path.abspath(input_video_dir)
  
  video_name = os.path.splitext(os.path.basename(input_video_dir))[0]

  conda_gvhmr = "conda run -n gvhmr --no-capture-output"
  ret = run(f"{conda_gvhmr} python tools/demo/demo.py --video={input_video_dir} -s", cwd=GVHMR_DIR)

  if ret != 0:
      print("GVHMR failed!")
      return

  gvhmr_result = os.path.join(GVHMR_DIR, "outputs", "demo", video_name, "hmr4d_results.pt")
  if not os.path.exists(gvhmr_result):
      print(f"Expected output not found: {gvhmr_result}")
      return

  save_path = os.path.join(GMR_DIR, "assets", "save_data", f"{video_name}.pkl")
  os.makedirs(os.path.dirname(save_path), exist_ok=True)

  conda_gmr = "conda run -n gmr --no-capture-output"
  run(
      f"{conda_gmr} python scripts/gvhmr_to_robot.py"
      f" --gvhmr_pred_file '{gvhmr_result}'"
      f" --robot unitree_g1"
      f" --record_video"
      f" --save_path '{save_path}'",
      cwd=GMR_DIR
  )

  print(f"\nDone! Output saved to: {save_path}")




if __name__ == "__main__":
  main()