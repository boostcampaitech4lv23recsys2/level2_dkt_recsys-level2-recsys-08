class Features:
    # FEAT = ['userID', 'assessmentItemID', 'testId', 
    #    'KnowledgeTag', 'user_correct_answer', 'user_total_answer', 'user_acc',
    #    'month', 'day', 'hour', 'big_category', 'problem_num',
    #    'mid_category', 'test_mean', 'test_std', 'test_sum', 'tag_mean',
    #    'tag_std', 'tag_sum', 'time_category', 'solvesec_3600']
    
    # CAT_FEAT = ['userID', 'assessmentItemID', 'testId', 
    #    'KnowledgeTag', 'month', 'day', 'hour', 'big_category', 'problem_num',
    #    'mid_category', 'time_category']
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
    FEAT = ['userID', 'assessmentItemID', 'testId',
       'KnowledgeTag', 
       'user_correct_answer', 'user_total_answer', 'user_acc', 'month',
       'hour', 'dayname', 'big_category', 'problem_num', 'mid_category',
       'test_mean', 'test_std', 'test_sum', 'tag_mean', 'tag_std', 'solvesec', 'solvesec_3600', 'time_category',
       'solvesec_cumsum', 'solvecumsum_category', 'big_category_acc',
       'big_category_std', 'big_category_cumconut', 'big_category_user_acc',
       'big_category_user_std', 'big_category_answer',
       'big_category_answer_log1p']

    CAT_FEAT = ['userID','assessmentItemID','testId','KnowledgeTag','big_category','mid_category',
                'problem_num','dayname','month','time_category','solvecumsum_category']