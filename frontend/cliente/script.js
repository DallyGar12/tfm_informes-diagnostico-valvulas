// ============================================================================
// DASHBOARD DE DIAGNÓSTICO
// ============================================================================

const API_URL = "http://localhost:5000/api";

let valvesData = [];
let selectedId = null;
let chart = null;

// ============================================================================
// Severidad
// ============================================================================

function severidadFriccion(med, rec) {
    const diff = Math.abs(med - rec);
    if (diff > 10) return "severo";
    if (diff > 5) return "moderado";
    return "normal";
}

function severidadCarga(med, rec) {
    if (rec === 0) return "normal";
    const pct = Math.abs((med / rec - 1) * 100);
    if (pct > 25) return "severo";
    if (pct > 10) return "moderado";
    return "normal";
}

function severidadBanda(med, rec) {
    if (rec === 0) return "normal";
    const pct = (med / rec) * 100;
    if (pct > 120) return "severo";
    if (pct > 100) return "moderado";
    return "normal";
}

function severidadLinealidad(med, rec) {
    if (rec === 0) return "normal";
    const pct = (med / rec) * 100;
    if (pct > 130) return "severo";
    if (pct > 100) return "moderado";
    return "normal";
}

const diagnosisMap = {
    'Fricción': { severo: '⚠️ Desgaste de packing crítico', moderado: '📌 Monitorear lubricación', normal: '✅ Normal' },
    'Carga en Asiento/Torque': { severo: '⚠️ Riesgo de fuga interna', moderado: '📌 Verificar tendencia', normal: '✅ Normal' },
    'Banda Muerta': { severo: '⚠️ Revisar posicionador urgente', moderado: '📌 Calibración necesaria', normal: '✅ Normal' },
    'Linealidad Dinámica': { severo: '⚠️ Problemas graves de control', moderado: '📌 Verificar caracterización', normal: '✅ Normal' }
};

// ============================================================================
// API
// ============================================================================

async function loadValves() {
    try {
        const res = await fetch(`${API_URL}/valvulas`);
        if (!res.ok) throw new Error('API error');
        valvesData = await res.json();
        console.log(`✅ ${valvesData.length} válvulas cargadas`);
        updateHeader();
        updateList();
    } catch (error) {
        console.error(error);
        document.getElementById('valve-list').innerHTML = 
            '<div style="padding:20px;text-align:center;color:#fb923c;">❌ Error conectando con API</div>';
    }
}

async function loadSummary() {
    try {
        const res = await fetch(`${API_URL}/resumen`);
        return await res.json();
    } catch {
        return null;
    }
}

async function loadDetail(id) {
    try {
        const res = await fetch(`${API_URL}/valvulas/${id}`);
        return await res.json();
    } catch {
        return null;
    }
}

// ============================================================================
// Render
// ============================================================================

async function updateHeader() {
    const summary = await loadSummary();
    if (summary) {
        document.getElementById('header-stats').innerHTML = `
            <div class="stat-badge severo"><span class="dot"></span>${summary.alerta || 0} ALERTA</div>
            <div class="stat-badge moderado"><span class="dot"></span>${summary.aceptable_com || 0} ACEPTABLE COM</div>
            <div class="stat-badge normal"><span class="dot"></span>${summary.aceptable || 0} ACEPTABLE</div>
            <div class="total-count"><strong>${summary.total || 0}</strong> válvulas</div>
        `;
    }
}

function updateList() {
    if (!valvesData.length) return;
    
    document.getElementById('valve-count').textContent = `${valvesData.length} válvulas diagnosticadas`;
    
    const container = document.getElementById('valve-list');
    container.innerHTML = valvesData.map(v => {
        let statusClass = "normal", statusText = "Normal";
        if (v.status === "ALERTA") { statusClass = "severo"; statusText = "ALERTA"; }
        else if (v.status === "ACEPTABLE CON COMENTARIOS") { statusClass = "moderado"; statusText = "ACEPTABLE COM"; }
        
        const measurements = v.measurements || [];
        const getMeas = (name) => measurements.find(m => m.name === name) || { measured: 0, recommended: 0 };
        
        const f = getMeas('Fricción');
        const c = getMeas('Carga en Asiento/Torque');
        const b = getMeas('Banda Muerta');
        const l = getMeas('Linealidad Dinámica');
        
        return `
            <button class="valve-item ${selectedId === v.id ? 'selected' : ''}" onclick="selectValve(${v.id})">
                <div class="valve-item-top">
                    <div>
                        <div class="valve-name">${v.name}</div>
                        <div class="valve-status">${v.status}</div>
                    </div>
                    <span class="badge ${statusClass}">${statusText}</span>
                </div>
                <div class="valve-dots">
                    <div class="valve-dot-item"><span class="dot-indicator ${severidadFriccion(f.measured, f.recommended)}"></span><span class="dot-label">Fri</span></div>
                    <div class="valve-dot-item"><span class="dot-indicator ${severidadCarga(c.measured, c.recommended)}"></span><span class="dot-label">Car</span></div>
                    <div class="valve-dot-item"><span class="dot-indicator ${severidadBanda(b.measured, b.recommended)}"></span><span class="dot-label">Ban</span></div>
                    <div class="valve-dot-item"><span class="dot-indicator ${severidadLinealidad(l.measured, l.recommended)}"></span><span class="dot-label">Lin</span></div>
                </div>
            </button>
        `;
    }).join('');
}

async function selectValve(id) {
    selectedId = id;
    updateList();
    
    const detail = await loadDetail(id);
    if (!detail) return;
    
    document.getElementById('empty-state').style.display = 'none';
    document.getElementById('detail-view').style.display = 'block';
    
    let statusClass = "normal", statusText = "Normal";
    if (detail.status === "ALERTA") { statusClass = "severo"; statusText = "ALERTA"; }
    else if (detail.status === "ACEPTABLE CON COMENTARIOS") { statusClass = "moderado"; statusText = "ACEPTABLE COM"; }
    
    document.getElementById('detail-name').textContent = `Válvula ${detail.name}`;
    document.getElementById('detail-status').textContent = detail.status;
    document.getElementById('detail-badge').innerHTML = `<span class="badge ${statusClass}">${statusText}</span>`;
    document.getElementById('detail-recommendation').textContent = detail.recommendation || "Sin recomendación";
    
    let action = '';
    if (detail.actionPlan) {
        if (typeof detail.actionPlan === 'object') {
            action = Object.values(detail.actionPlan).filter(v => v && !v.includes('✅')).join(' | ');
        } else {
            action = detail.actionPlan;
        }
    }
    document.getElementById('detail-action').textContent = action || "Monitoreo programado";
    
    renderChart(detail);
    renderParamCards(detail);
}

function renderChart(detail) {
    const canvas = document.getElementById('chart');
    if (chart) { chart.destroy(); }
    
    const measurements = detail.measurements || [];
    const labels = ['Fricción', 'Carga', 'Banda Muerta', 'Linealidad'];
    const measured = [0, 0, 0, 0];
    const recommended = [0, 0, 0, 0];
    
    measurements.forEach(m => {
        if (m.name === 'Fricción') { measured[0] = m.measured; recommended[0] = m.recommended; }
        else if (m.name === 'Carga en Asiento/Torque') { measured[1] = m.measured; recommended[1] = m.recommended; }
        else if (m.name === 'Banda Muerta') { measured[2] = m.measured; recommended[2] = m.recommended; }
        else if (m.name === 'Linealidad Dinámica') { measured[3] = m.measured; recommended[3] = m.recommended; }
    });
    
    chart = new Chart(canvas, {
        type: 'bar',
        data: {
            labels,
            datasets: [
                { label: 'Medido', data: measured, backgroundColor: '#22d3ee', borderRadius: 4 },
                { label: 'Recomendado', data: recommended, backgroundColor: '#4ade80', borderRadius: 4 }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { position: 'bottom', labels: { color: '#f1f5f9' } } },
            scales: { y: { ticks: { color: '#f1f5f9' }, grid: { color: '#2a3145' } },
                      x: { ticks: { color: '#f1f5f9' }, grid: { color: '#2a3145' } } }
        }
    });
}

function renderParamCards(detail) {
    const measurements = detail.measurements || [];
    
    const cards = measurements.map(m => {
        let severity = m.severity || 'normal';
        if (m.name === 'Fricción') severity = severidadFriccion(m.measured, m.recommended);
        else if (m.name === 'Carga en Asiento/Torque') severity = severidadCarga(m.measured, m.recommended);
        else if (m.name === 'Banda Muerta') severity = severidadBanda(m.measured, m.recommended);
        else if (m.name === 'Linealidad Dinámica') severity = severidadLinealidad(m.measured, m.recommended);
        
        const severityText = { severo: 'Severo', moderado: 'Moderado', normal: 'Normal' }[severity];
        const diagnosis = diagnosisMap[m.name]?.[severity] || 'Sin diagnóstico';
        
        return `
            <div class="param-card ${severity}">
                <div class="param-top">
                    <span class="param-name">${m.name}</span>
                    <span class="badge outline-${severity}">${severityText}</span>
                </div>
                <div class="param-diag">${diagnosis}</div>
                <div class="param-values">
                    <div class="param-value-box">
                        <span class="param-value-label">Medido</span>
                        <span class="param-value-num value-measured">${m.measured.toFixed(2)}<span class="param-value-unit">${m.unit || 'unid'}</span></span>
                    </div>
                    <div class="param-value-box">
                        <span class="param-value-label">Recomendado</span>
                        <span class="param-value-num value-recommended">${m.recommended.toFixed(2)}<span class="param-value-unit">${m.unit || 'unid'}</span></span>
                    </div>
                    <div class="param-value-box">
                        <span class="param-value-label">Desviación</span>
                        <span class="param-value-num value-deviation-${severity}">${Math.abs(m.measured - m.recommended).toFixed(2)}<span class="param-value-unit">${m.unit || 'unid'}</span></span>
                    </div>
                </div>
            </div>
        `;
    }).join('');
    
    document.getElementById('param-cards').innerHTML = cards;
}

window.selectValve = selectValve;

// Inicializar
loadValves();