const assert=require('node:assert/strict');
const core=require('../embedding_feature_audit/assets/enhancements.js');
assert.equal(core.scoreGain({balanced_accuracy:.8},{balanced_accuracy:.5},'categorical'),.8-.5);
assert.equal(core.scoreGain({mae:2},{mae:4},'continuous'),.5);
assert.equal(core.scoreGain({mae:2},{mae:0},'continuous'),null);
assert.ok(core.gapDescription([{organization_gap:0},{organization_gap:.3},{organization_gap:null}]).includes('1/2'));
assert.ok(core.gapDescription([]).includes('No estimable'));
console.log('Enhancement summary numerical checks passed.');
