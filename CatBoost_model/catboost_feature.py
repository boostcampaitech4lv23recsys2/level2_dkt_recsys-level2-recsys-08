class Features:
    # FEAT = ['userID', 'assessmentItemID', 'testId', 
    #    'KnowledgeTag', 'user_correct_answer', 'user_total_answer', 'user_acc',
    #    'month', 'day', 'hour', 'big_category', 'problem_num',
    #    'mid_category', 'test_mean', 'test_std', 'test_sum', 'tag_mean',
    #    'tag_std', 'tag_sum', 'time_category', 'solvesec_3600']
    
    # CAT_FEAT = ['userID', 'assessmentItemID', 'testId', 
    #    'KnowledgeTag', 'month', 'day', 'hour', 'big_category', 'problem_num',
    #    'mid_category', 'time_category']

    # 27 개
    # FEAT = ['userID',
    #      'assessmentItemID',
    #      'testId',
    #      'KnowledgeTag',
    #      'user_correct_answer',
    #      'user_total_answer',
    #      'big_category',
    #      'mid_category',
    #      'problem_num',
    #      'month','dayname','hour',
    #     #  'day',
    #      'user_acc',
    #      'test_mean',
    #      'test_sum',
    #      'test_std',
    #      'tag_std',
    #      'tag_mean',
    #     #  'tag_sum',
    #      'solvesec_3600',
    #      'time_category',
    #      'solvesec_cumsum',
    #      'solvecumsum_category',
    #      'big_category_acc',
    #      'big_category_std',
    #      'big_category_cumconut'
    #     ]

    # ====================================================================================
    # 30개
    # FEAT = ['userID', 'assessmentItemID', 'testId',
    #    'KnowledgeTag', 
    #    'user_correct_answer', 'user_total_answer', 'user_acc', 'month',
    #    'hour', 
    #   'dayname', 
    #    'big_category', 'problem_num', 'mid_category',
    #   #  'test_mean', 
    #   #  'test_std', 
    #   #  'test_sum', 
    #   #  'tag_mean', 'tag_std', 
    #    'solvesec', 
    #   #  'solvesec_3600', 
    #    'time_category',
    #   #  'solvesec_cumsum', 
    #   #  'solvecumsum_category',
    #     'big_category_acc',
    #    'big_category_std', 'big_category_cumconut', 
    #    'big_category_user_acc',
    #    'big_category_user_std', 'big_category_answer',
    #    'big_category_answer_log1p',
    #    #-------------------------------
    #   #  'correctRatio__by_user',
    #     # 'correctRatio__by_test_paper',
    #   #  'correctRatio__by_tag', 
    #    'correctRatio__by_prob' 
    #   # 'correctRatio__by_BC',
    #   #  'correctRatio__by_TC'
    #    ]
       # ====================================================================================

   # 35 개
    # FEAT = ['userID', 'assessmentItemID', 'testId',
    #   'KnowledgeTag',
    #   'user_correct_answer', 'user_total_answer', 'user_acc', 'month',
    #   'hour', 'dayname', 'big_category', 'problem_num', 'mid_category',
    #   'test_mean', 'test_std', 'test_sum', 'tag_mean', 'tag_std', 'solvesec', 'solvesec_3600', 
    #   'time_category','solvesec_cumsum', 'solvecumsum_category', 
    #   'big_category_acc',
    #   'big_category_std', 'big_category_cumconut', 'big_category_user_acc',
    #   'big_category_user_std', 'big_category_answer',
    #   'big_category_answer_log1p',
    #   # 'prob_elapsed_time', 
    #   # 'normalized_p_elapsed_time',
    #   'user_prob_elapsed_time_aver', 'test_prob_elapsed_time_aver',
    #   'BC_prob_elapsed_time_aver', 'MC_prob_elapsed_time_aver',
    #   'normalized_solvesec']

    FEAT = ['userID', 
      'assessmentItemID',
       'testId',
       'KnowledgeTag',
        'big_category',
        'mid_category',
        'problem_num',
        'month',
        'day',
        # 'solvesec',
        'test_solvesec',
        'dayname',
       'hour',
       'user_correct_answer',
        'user_total_answer',
        'user_acc',
        'user_BC_correct_answer',
        'user_BC_total_answer',
        'user_BC_acc',
        'user_testId_correct_answer',
        'user_testId_total_answer',
        'user_testId_acc',
                
        'correctRatio__by_test_paper',
       'correctRatio__by_prob', 
      'correctRatio__by_BC'
        ]

    CAT_FEAT = ['userID','assessmentItemID','testId',
                'KnowledgeTag',
                'big_category','mid_category',
               'problem_num',
               'dayname',
               'month',
               'day',
               'dayname',
               'hour'

              #  'time_category'
              #  ,'solvecumsum_category'
               ]