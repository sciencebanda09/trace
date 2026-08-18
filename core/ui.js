export function setExclusiveView(views,id){views.forEach(view=>view.classList.toggle('active',view.id===id))}
