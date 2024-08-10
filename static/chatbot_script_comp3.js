
    (function() {
        var chatbotContainer = document.createElement('div');
        chatbotContainer.id = 'chatbot-container';
        chatbotContainer.style.position = 'fixed';
        chatbotContainer.style.bottom = '20px';
        chatbotContainer.style.right = '20px';
        chatbotContainer.style.width = '300px';
        chatbotContainer.style.height = '400px';
        chatbotContainer.style.border = '1px solid #ccc';
        chatbotContainer.style.borderRadius = '10px';
        chatbotContainer.style.overflow = 'hidden';
        chatbotContainer.style.display = 'flex';
        chatbotContainer.style.flexDirection = 'column';

        var chatHeader = document.createElement('div');
        chatHeader.style.padding = '10px';
        chatHeader.style.backgroundColor = '#f1f1f1';
        chatHeader.style.borderBottom = '1px solid #ccc';
        chatHeader.innerHTML = '<h3 style="margin: 0;">Chatbot</h3>';

        var chatMessages = document.createElement('div');
        chatMessages.style.flexGrow = '1';
        chatMessages.style.overflowY = 'auto';
        chatMessages.style.padding = '10px';

        var chatInput = document.createElement('input');
        chatInput.type = 'text';
        chatInput.placeholder = 'Type your message...';
        chatInput.style.width = '100%';
        chatInput.style.padding = '10px';
        chatInput.style.border = 'none';
        chatInput.style.borderTop = '1px solid #ccc';

        chatbotContainer.appendChild(chatHeader);
        chatbotContainer.appendChild(chatMessages);
        chatbotContainer.appendChild(chatInput);

        document.body.appendChild(chatbotContainer);

        chatInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                var message = this.value;
                if (message.trim() !== '') {
                    addMessage('You', message);
                    sendMessage(message);
                    this.value = '';
                }
            }
        });

        function addMessage(sender, message) {
            var messageElement = document.createElement('p');
            messageElement.innerHTML = '<strong>' + sender + ':</strong> ' + message;
            chatMessages.appendChild(messageElement);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        function sendMessage(message) {
            fetch('http://localhost:8000/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    company_id: 'comp3',
                    message: message
                })
            })
            .then(response => {
                if (!response.ok) {
                    return response.text().then(text => { throw new Error(text) });
                }
                return response.json();
            })
            .then(data => {
                addMessage('Chatbot', data.response);
            })
            .catch((error) => {
                console.error('Error:', error);
                addMessage('Chatbot', 'Error: ' + error.message);
            });
        }
    })();
    