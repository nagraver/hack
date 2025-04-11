async function sendData() {
    const input1 = document.getElementById('input1').value;
    const input2 = document.getElementById('input2').value;
    
    
    const response = await fetch('http://localhost:5000/calculate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            input1: parseFloat(input1),
            input2: parseFloat(input2)
        })
    });
    
    const data = await response.json();
    document.getElementById('result').textContent = data.result;
}