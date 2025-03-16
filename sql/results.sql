-- quick little query
select id,
       (model_name || ':' || model_version)               as 'mid',
       description,
       strftime('%Y-%m-%d %H:%M:%S', run_start_timestamp) as 'start',
       dataset_name,
       fit_epochs,
       learning_rate,
       batch_size,
       backend,
       backend_proc,
       eval_accuracy
from model_test_run
order by run_start_timestamp