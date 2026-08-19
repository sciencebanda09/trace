export function setExclusiveView(views,id){views.forEach(view=>view.classList.toggle('active',view.id===id))}

export function renderDatasetRows(tbody, rows, { titleCase }) {
  const fragment = document.createDocumentFragment();
  rows.forEach(row => {
    const tr = document.createElement('tr');
    const cell = value => {
      const td = document.createElement('td');
      td.textContent = String(value ?? '');
      tr.append(td);
      return td;
    };
    const sourceCell = cell('');
    const source = document.createElement('span');
    source.className = `source-tag ${row.source === 'live' ? 'live' : 'seed'}`;
    source.textContent = String(row.id ?? 'unknown');
    sourceCell.append(source);
    ['occlusion', 'lighting', 'orientation', 'environment'].forEach(key => cell(titleCase(row[key])));
    const result = row.result === 'success' ? 'success' : 'failure';
    const resultCell = cell('');
    const status = document.createElement('span');
    status.className = `status-badge ${result}`;
    status.textContent = result.toUpperCase();
    resultCell.append(status);
    cell(row.recovery === 'yes' ? 'Yes' : 'No');
    const dateCell = cell(new Date(row.created).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }));
    dateCell.className = 'font-mono dataset-date-cell';
    const actionCell = cell('');
    actionCell.className = 'align-right';
    const removeButton = document.createElement('button');
    removeButton.className = 'action-icon-btn';
    removeButton.type = 'button';
    removeButton.title = 'Delete record';
    removeButton.dataset.removeRecord = String(row.id ?? '');
    removeButton.setAttribute('aria-label', `Delete record ${String(row.id ?? '')}`);
    removeButton.textContent = '×';
    actionCell.append(removeButton);
    fragment.append(tr);
  });
  tbody.replaceChildren(fragment);
}
