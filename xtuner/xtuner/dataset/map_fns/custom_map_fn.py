# Copyright (c) OpenMMLab. All rights reserved.


# def alpaca_map_fn(example):
#     if example.get('output') == '<nooutput>':
#         return {'conversation': []}
#     else:
#         return {
#             'conversation': [{
#                 'input': f"{example['instruction']}\n{example['input']}",
#                 'output': example['output']
#             }]
#         }
# Suppose the function is stored in ./map_fn.py
SYSTEM_ALPACA = ('Below is an instruction that describes a task. '
                 'Write a response that appropriately completes the request.\n')
def alpaca_map_fn(example):
    if example.get('output') == '<nooutput>':
        return {'conversation': []}
    else:
        return {
            'conversation': [{
                'system': SYSTEM_ALPACA,
                'input': f"{example['instruction']}",
                'output': example['output']
            }]
        }