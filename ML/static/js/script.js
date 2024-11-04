const socket = io.connect('http://localhost:5000');

socket.on('command', function(data) {
    const arrows = {
        'up': document.getElementById('arrow-up'),
        'down': document.getElementById('arrow-down'),
        'left': document.getElementById('arrow-left'),
        'right': document.getElementById('arrow-right')
    };

    for (let key in arrows) {
        arrows[key].classList.remove('active');
    }

    if (data.direction in arrows) {
        arrows[data.direction].classList.add('active');
    }
});

document.getElementById('toggle-camera').addEventListener('click', function() {
    socket.emit('toggle_camera');
});

document.getElementById('sensitivity').addEventListener('input', function() {
    const sensitivity = this.value;
    socket.emit('update_sensitivity', { sensitivity: sensitivity });
});


// Função para iniciar a coleta de dados
function startDataCollection(label) {
    socket.emit('start_collection', { label: label });
    document.getElementById('status').innerText = `Coletando dados para: ${label}`;
    setTimeout(() => {
        socket.emit('stop_collection');
        document.getElementById('status').innerText = "Coleta de dados finalizada.";
    }, 5000); // Coleta por 5 segundos
}

// Eventos dos botões de coleta
document.querySelectorAll('.collect-btn').forEach(button => {
    button.addEventListener('click', () => {
        const label = button.getAttribute('data-label');
        startDataCollection(label);
    });
});

// Evento para treinar o modelo
document.getElementById('train-model').addEventListener('click', () => {
    socket.emit('train_model');
    document.getElementById('status').innerText = "Treinando modelo...";
});

// Atualização do feedback visual para os comandos
socket.on('command', function(data) {
    const arrows = {
        'up': document.getElementById('arrow-up'),
        'down': document.getElementById('arrow-down'),
        'left': document.getElementById('arrow-left'),
        'right': document.getElementById('arrow-right')
    };

    for (let key in arrows) {
        arrows[key].classList.remove('active');
    }

    if (data.direction in arrows) {
        arrows[data.direction].classList.add('active');
    }
});

// Controle de sensibilidade
document.getElementById('sensitivity').addEventListener('input', function() {
    const sensitivity = this.value;
    socket.emit('update_sensitivity', { sensitivity: sensitivity });
});
