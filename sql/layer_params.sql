-- Nice list of layers and params
select mtr.id as 'test run id',
       mtr.run_start_timestamp,
       l.id   as 'layer_id',
       l.layer_type_name,
       hp.id  as 'param_id',
       hp.var_name,
       hp.var_value
from model_test_run as mtr
         left join model_layer as l on l.model_test_run_id = mtr.id
         left join hyperparameter as hp on l.id = hp.model_layer_id
where
    mtr.id > 0
--     l.model_test_run_id = 3
--    mtr.id = 3
--  and
--     hp.var_name == 'activation'
--    hp.var_name == 'batch_shape'
--   and hp.var_value == 'relu'
order by mtr.id, l.sequence_number, hp.var_name


