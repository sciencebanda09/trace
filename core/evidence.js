export function confidenceLabel(value){return value>=.8?'high':value>=.6?'medium':'low'}
export function calibrationCorrection({attribute,from,to,evidence}){return{id:`correction-${crypto.randomUUID?.()||Date.now()}`,created:Date.now(),attribute,from,to,evidence}}
