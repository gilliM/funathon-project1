# %%
import sys
import os

# Append parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shutil
import subprocess 

from utils import upload_file_s3

input_script = "intermediate_solutions/3_RF.py"
model_name = "rf_model_final"
s3_name = "rf_model_test.joblib"

# Step 1: Copy 3_RF.py to temp.py
shutil.copy(input_script, "temp/temp.py")

# Step 2: Add lines to temp script to dump model 
with open("temp/temp.py", "a") as f:
    f.write("\n\n# Save model\n")
    f.write("import joblib\n")
    f.write(f"joblib.dump({model_name}, 'temp/temp.joblib')\n")

# Step 3: Run temp.py
subprocess.run(["uv", "run", "temp/temp.py"])

# Step 4: Upload temp.joblib to S3
upload_file_s3("temp/temp.joblib", s3_name)

# Step 5: Cleanup temporary files
os.remove("temp/temp.py")
os.remove("temp/temp.joblib")

# %%
