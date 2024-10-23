# 使用者操作手冊
舊的功能就不再贅述

## AI 功能敘述
### 1. SAM Everything (ai_everything)
- 使用方法：
    - step1:    點擊 Enable Everything
    - step2:　點擊 SAM Everything
    - step3:　選擇 GPU，若只有一顆則無須選擇
    - step4:　開始標記，在圖上選取任意範圍，總共須點擊兩次。因為此模式算法比較花時間，因此要稍等片刻才能輸入 Label 名稱（必填）、Group ID（選填）、 Label description （選填）
- 用途：
    - 適合大範圍（整張圖），最一開始的標記，可以一次標記大量零組件
    - 若用在中範圍會較為準確，漏標數少
- 優點：可以無須點擊零組件本身，可以任意選取範圍，並且範圍大
- 缺點：較不精準，可能會漏掉一些零組件未標記，需要搭配其他 AI 功能做完整標記
### 2. ai_polygon
- 使用方法：　滑鼠點擊單個零組件後 `ctrl + click left` ，輸入 Label 名稱（必填）、Group ID（選填）、 Label description （選填）
- 用途：適合範圍小、零組件須單一個別標記時
- 優點：最為精準

### 3. ai_mask
- 使用方法：　滑鼠點擊單個零組件後可持續點擊其他零組件，模型會將範圍內相同的零組件標記起來，按 `ctrl + click left` ，輸入 Label 名稱（必填）、Group ID（選填）、 Label description （選填）
- 用途：適合中範圍、零組件集中但須多個標記時
- 優點：比起單個標記、速度快很多

### 4. ai_boundingbox
- 使用方法：　與 ai_mask 相同，差別為標記完是矩形標注
- 用途：適合中範圍、零組件集中但須多個標記時
- 優點：比起單個標記、速度快很多

## How to develop

```bash
git clone https://github.com/jerrylin0829/labelme.git
cd labelme

# Install anaconda3 
curl -L https://github.com/wkentaro/dotfiles/raw/main/local/bin/install_anaconda3.sh | bash -s .
source .anaconda3/bin/activate

#Install labelme & set CUDA option
python install_labelme.py --cuda_ver=12.4 --cuda_dir=/path/to/cuda

```
- `--cuda_ver`: (Optional) Specify the CUDA version you want to use, e.g., ``--cuda_ver=12.4.`` If omitted, the script will attempt to detect the installed CUDA version automatically.
- `--cuda_dir`: (Optional) Specify the custom directory where CUDA is installed, e.g., `--cuda_dir=/usr/local/cuda-12`. If omitted, the script will use the default /usr/local path to find CUDA installations.

### How to build standalone executable

Below shows how to build the standalone executable on macOS, Linux and Windows.  

```bash
# Setup conda
conda create --name labelme python=3.9 # or 3.10
conda activate labelme

# Build the standalone executable
pip install .
pip install 'matplotlib<3.3'
pip install pyinstaller
pyinstaller labelme.spec
dist/labelme --version
```