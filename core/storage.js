const DB='trace-store',STORE='clips';
export function openTraceDB(){return new Promise((resolve,reject)=>{const r=indexedDB.open(DB,1);r.onupgradeneeded=()=>{if(!r.result.objectStoreNames.contains(STORE))r.result.createObjectStore(STORE,{keyPath:'id'})};r.onsuccess=()=>resolve(r.result);r.onerror=()=>reject(r.error)})}
export async function getRecords(){const db=await openTraceDB();return new Promise((res,rej)=>{const q=db.transaction(STORE).objectStore(STORE).getAll();q.onsuccess=()=>res(q.result);q.onerror=()=>rej(q.error)})}
export async function putRecord(row){const db=await openTraceDB();return new Promise((res,rej)=>{const q=db.transaction(STORE,'readwrite').objectStore(STORE).put(row);q.onsuccess=res;q.onerror=()=>rej(q.error)})}
export async function removeRecord(id){const db=await openTraceDB();return new Promise((res,rej)=>{const q=db.transaction(STORE,'readwrite').objectStore(STORE).delete(id);q.onsuccess=res;q.onerror=()=>rej(q.error)})}
export async function clearRecords(){const db=await openTraceDB();return new Promise((res,rej)=>{const q=db.transaction(STORE,'readwrite').objectStore(STORE).clear();q.onsuccess=res;q.onerror=()=>rej(q.error)})}
export async function storageEstimate(){return navigator.storage?.estimate?navigator.storage.estimate():{usage:0,quota:0}}
