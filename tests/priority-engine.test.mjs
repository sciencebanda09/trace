import test from 'node:test';
import assert from 'node:assert/strict';
import { TRACE_SCHEMA, DEFAULT_WEIGHTS } from '../core/schema.js';
import { scoreExperiments, calculateDatasetReadiness } from '../core/priority-engine.js';
const clips=[
 {occlusion:'none',lighting:'normal',orientation:'upright',environment:'bench',result:'success',recovery:'no'},
 {occlusion:'none',lighting:'normal',orientation:'upright',environment:'bench',result:'success',recovery:'no'},
 {occlusion:'partial',lighting:'low-light',orientation:'rotated',environment:'bench',result:'failure',recovery:'no'}
];
test('ranking is deterministic',()=>assert.deepEqual(scoreExperiments({clips,schema:TRACE_SCHEMA,weights:DEFAULT_WEIGHTS}),scoreExperiments({clips,schema:TRACE_SCHEMA,weights:DEFAULT_WEIGHTS})));
test('failure-correlated dimensions reach the top recommendation',()=>{const ranked=scoreExperiments({clips,schema:TRACE_SCHEMA,weights:DEFAULT_WEIGHTS});assert.equal(ranked[0].occlusion,'partial');assert.ok(ranked[0].failure>0)});
test('readiness remains bounded and reacts to added coverage',()=>{const before=calculateDatasetReadiness(clips,TRACE_SCHEMA);const after=calculateDatasetReadiness([...clips,{occlusion:'heavy',lighting:'bright',orientation:'inverted',environment:'floor',result:'failure',recovery:'yes'}],TRACE_SCHEMA);assert.ok(before>=0&&after<=100);assert.ok(after>before)});
