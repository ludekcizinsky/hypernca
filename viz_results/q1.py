"""Get Table Results For Conditional Diffusion Models"""

import pandas as pd
import numpy as np


df = pd.read_json('results_stored/cond_diff_results.json')


num_diffusion_steps = 100
num_steps = 50
df_subset = df[(df['num_diffusion_steps']==num_diffusion_steps) & (df['num_steps']==num_steps)]



df_nca_diff = df_subset[df_subset['compare_type'] == 'nca_diffusion']
df_nca_diff.drop(columns=['kid_std','compare_type','num_diffusion_steps',
                          'num_steps'], inplace=True)
df_nca_diff.rename(columns={'kid_mean':'KID',
                            'lpips':'LPIPS',
                            'fid':'FID',
                            'PSNR':'PSNR',
                            'model_type':'Cond Model'}, inplace=True)
print(df_nca_diff.to_latex(float_format='%.3f',index=False))


df_gt_diff = df_subset[df_subset['compare_type'] == 'gt_diffusion']
df_gt_diff.drop(columns=['kid_std','compare_type','num_diffusion_steps',
                          'num_steps'], inplace=True)
df_gt_diff.rename(columns={'kid_mean':'KID',
                            'lpips':'LPIPS',
                            'fid':'FID',
                            'PSNR':'PSNR',
                            'model_type':'Cond Model'}, inplace=True)
print(df_gt_diff.to_latex(float_format='%.3f',index=False))



