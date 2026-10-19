/* One metric vocabulary and color rule shared by both report workspaces. */
(function(){
'use strict';
const definitions={
 r2:['R²','up'],mae:['MAE','relative_error'],spearman:['Spearman correlation','up'],
 balanced_accuracy:['Balanced accuracy','up'],macro_f1:['Macro F1','up'],micro_f1:['Micro F1','up'],accuracy:['Accuracy','up'],
 roc_auc:['ROC AUC','up'],average_precision:['Average precision','up'],subset_accuracy:['Exact-set accuracy','up'],
 entropy_delta:['Predicted − true entropy','zero'],normalized_entropy_delta:['Predicted − true normalized entropy','zero'],
 largest_label_share_delta:['Predicted − true largest label share','zero'],n_labels_delta:['Predicted − true label count','zero'],
 jensen_shannon_divergence:['Prediction/truth JS divergence','down'],mean_label_cardinality_delta:['Predicted − true tag cardinality','zero'],
 macro_prevalence_absolute_error:['Mean tag prevalence error','down']
};
const finite=x=>typeof x==='number'&&Number.isFinite(x);
function available(records){return Object.keys(definitions).filter(key=>records.some(row=>finite(row[key])));}
function gain(value,baseline,key){
 if(!finite(value)||!finite(baseline))return null;
 const rule=definitions[key][1];
 return rule==='relative_error'?(baseline>0?(baseline-value)/baseline:null):rule==='zero'?Math.abs(baseline)-Math.abs(value):rule==='down'?baseline-value:value-baseline;
}
function referenceText(key){const rule=definitions[key][1];return rule==='relative_error'?'relative error improvement':rule==='zero'?'improvement toward zero':rule==='down'?'baseline minus score':'score minus baseline';}
const api={definitions,available,gain,referenceText};
if(typeof module!=='undefined'&&module.exports)module.exports=api;else window.AuditMetrics=api;
})();
