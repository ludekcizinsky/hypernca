"""Plot The Metrics For H1 Experiments"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_json('results_stored/h1_df_real.json')


colors = ['blueviolet','deeppink','yellowgreen','tomato','black']
metric = 'lpips' # 'fid','psnr','lpips','kid_mean'
fontsize = 14


metric_name_dict = {
    'fid':'FID',
    'psnr':'PSNR',
    'lpips':'LPIPS',
    'kid_mean':'KID',
}


group_name_dict = {
    'nca_training_from_random_2025-05-03_23':'Random Weights',
    'traing_from_bubbly_weights_2025-05-04_13':'Bubbly Weights',
    'training_from_diffusion_sampled_weights_2025-05-04_13':'Unconditional Diffusion Weights (T=50)',
    'training_from_diffusion_sampled_weights_2025-05-04_13_26':'Unconditional Diffusion Weights (T=500)',
    'training_from_diffusion_sampled_weights_2025-05-18_12_16':'Conditional Diffusion Weights (T=50)',
}




def find_threshold_epochs(df, metric, threshold, direction='below'):
    """
    Returns a DataFrame with the first epoch per group where `metric` crosses `threshold`.
    `direction` can be 'below' or 'above'.
    """
    result = []
    
    for group_key, readable_name in group_name_dict.items():
        group_df = df[df['group_name'] == group_key]
        if direction == 'below':
            filtered = group_df[group_df[metric] < threshold]
        elif direction == 'above':
            filtered = group_df[group_df[metric] > threshold]
        else:
            raise ValueError("direction must be 'below' or 'above'")
        
        if not filtered.empty:
            first_hit = filtered.sort_values('epoch').iloc[0]
            result.append({
                'Group': readable_name,
                'Epoch': int(first_hit['epoch']),
                metric_name_dict[metric]: round(first_hit[metric], 3)
            })
        else:
            result.append({
                'Group': readable_name,
                'Epoch': None,
                metric_name_dict[metric]: None
            })
    
    return pd.DataFrame(result)

# Create threshold tables
# metric = 'fid'
# df_out = find_threshold_epochs(df, metric=metric, threshold=270, direction='below')
# df_out['Time'] = df_out['Epoch']*7.5 # 7.5 minutes per epoch on average

# df_out.loc[df_out['Group']=='Conditional Diffusion Weights (T=50)','Time'] += 10
# df_out.loc[df_out['Group']=='Unconditional Diffusion Weights (T=500)','Time'] += 10
# df_out.loc[df_out['Group']=='Unconditional Diffusion Weights (T=50)','Time'] += 10

# df_out.drop(columns=[metric_name_dict[metric]], inplace=True)

# print(df_out.to_latex(index=False, float_format='%.1f'))




for i,group in enumerate(df['group_name'].unique()):
    group_df = df[df['group_name'] == group][[metric,'epoch']]
    plt.plot(group_df['epoch'], group_df[metric], label=group_name_dict[group], color=colors[i])



plt.xlabel('Epoch', fontsize=fontsize)
plt.ylabel(metric_name_dict[metric], fontsize=fontsize)
plt.title(f'{metric_name_dict[metric]} as a function NCA training epochs',fontsize=fontsize+2, fontweight='bold')
plt.legend()
plt.grid()
plt.savefig(f'{metric}_training_epochs.png', dpi=300, bbox_inches='tight')
#plt.show()



