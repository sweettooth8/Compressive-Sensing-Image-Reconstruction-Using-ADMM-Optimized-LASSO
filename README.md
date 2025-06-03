4. # Compressive Sensing Image Reconstruction Using ADMM Optimized LASSO
   This is the final project of SUSTech's Postgraduate Machine Learning Spring 2025. You can find the course's information at https://fangkongx.github.io/Teaching/MAT8034/Spring2025/index.html. 
   
   For detailed information about the project, see the pdf included in the respository:  `Slides.pdf`.
   
   ## Project Structure
   
   - `Image/`: Includes the original images for reconstruction.
   - `Output/`: For storage of reconstructed images.
   - `admm_lasso.py`: Implements the basic ADMM solving LASSO problem.
   - `admm_lasso_torch.py`: `admm_lasso.py`'s torch version.
   - `admm_lasso_module.py`: Encapsulate `admm_lasso_torch.py` as a `nn.module` class. 
   - `admm_lasso_batch_module.py`: `admm_lasso_module.py`'s batch version, which can batch process the input parameters.
   - `admm_lasso_sparse.py`: This is the demo of ADMM_LASSO specialized for computing sparse matrices.
   - `example.py`: This is the example based on Sklearn's LASSO to show why we choose LASSO.
   - `compressive_sensing.py`: Provides a practical application of ADMM_LASSO to compressive sensing, including signal generation, measurement simulation, and reconstruction evaluation.
   - `Slides.pdf`: Introduces the project goal, mathematical pipeline, experiemental results, and discussion of results.
   - `requirements.txt`: The requirements for the project.
   
   ## Requirements
   
   - Python 3.7+
   - PyTorch (with CUDA support for GPU acceleration)
   - SciPy (for median filtering)
   - scikit-learn (for KFold cross-validation)
   - tqdm (for progress bars)
   
   ## Installation
   
   1. Git clone repository.
   
      ```
      git clone https://github.com/sweettooth8/Compressive-Sensing-Image-Reconstruction-Using-ADMM-Optimized-LASSO.git
      ```
   
   2. Install the `requirements.txt` file.
   
      ```bash
      pip install -r requirements.txt
      ```
   
   ## Quick Start
   
   1. **Prepare the Image**: Place the image to be recovered (e.g., `nature.bmp`) in a path accessible by the project, by default `Image/nature.bmp`. You can modify the `img_path` parameter in the `ImageRecover` class within `compressive_sensing.py` to specify a different image.
   2. **Configure Parameters (Optional)**:
      - In the `compressive_sensing.py` file, you can modify the parameters during the instantiation of the `ImageRecover` class:
        - `img_path`: Path to the input image.
        - `block_size`: Size of the image blocks (e.g., `8` for 8x8 blocks).
        - `S_values`: A NumPy array defining the number of sampled pixels in each block (or different sampling rate configurations). The script will sequentially perform recovery for each `S` value in the array.
        - `alpha_val_list`: A list of candidate $\alpha$ values for cross-validation.
        - `num_cv_folds`: Number of folds for cross-validation.
        - `verbose`: Whether to print detailed ADMM iteration information.
   
   3. **Run the Script**:
   
   ```bash
   python compressive_sensing.py
   ```
   
   Modify parameters directly within the script to experiment with different settings, such as sparsity levels, noise conditions, and algorithm iterations.
   
   4. **View Results**:
   
      - Recovered images will be saved in the `Output` directory (or another directory specified by `RESULTS_DIR`).
      - Histograms of the optimal $\alpha$ parameter distribution for each `S` value will also be saved in this directory.
      - The console will output the mean $\alpha$, standard deviation, and total runtime for each `S` value.
   
      
