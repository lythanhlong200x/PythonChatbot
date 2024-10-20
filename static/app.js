class Chatbox {
    constructor() {
        this.args = {
            openButton: document.querySelector('.chatbox__button'),
            chatBox: document.querySelector('.chatbox__support'),
            sendButton: document.querySelector('.send__button'),
            header: document.querySelector('.chatbox__content--header')  // Thêm phần header để thay đổi giao diện
        }
        this.state = false;
        this.messages = [];
        this.currentBotMode = 'support';  // Mặc định là chatbot hỗ trợ
    }

    display() {
        const { openButton, chatBox, sendButton } = this.args;

        openButton.addEventListener('click', () => this.toggleState(chatBox));
        sendButton.addEventListener('click', () => this.onSendButton(chatBox));

        const node = chatBox.querySelector('input');
        node.addEventListener("keyup", ({ key }) => {
            if (key === "Enter") {
                this.onSendButton(chatBox);
            }
        });
    }

    toggleState(chatbox) {
        this.state = !this.state;
        if (this.state) {
            chatbox.classList.add('chatbox--active');
        } else {
            chatbox.classList.remove('chatbox--active');
        }
    }

    onSendButton(chatbox) {
        var textField = chatbox.querySelector('input');
        let text1 = textField.value;
        if (text1 === "") {
            return;
        }

        let msg1 = { name: "User", message: text1 };
        this.messages.push(msg1);

        fetch($SCRIPT_ROOT + '/predict', {
            method: 'POST',
            body: JSON.stringify({ message: text1 }),
            mode: 'cors',
            headers: {
                'Content-Type': 'application/json'
            },
        })
            .then(r => r.json())
            .then(r => {
                let msg2 = { name: "Sam", message: r.answer };

                // Kiểm tra xem có cần chuyển chế độ bot không
                if (r.answer.includes("AI Bot")) {
                    this.switchToAiBot();
                } else if (r.answer.includes("Support Bot")) {
                    this.switchToSupportBot();
                }

                this.messages.push(msg2);
                this.updateChatText(chatbox);
                textField.value = '';
            }).catch((error) => {
                console.error('Error:', error);
                this.updateChatText(chatbox);
                textField.value = '';
            });
    }

    updateChatText(chatbox) {
        var html = '';
        this.messages.slice().reverse().forEach(function (item) {
            if (item.name === "Sam") {
                html += '<div class="messages__item messages__item--visitor">' + item.message + '</div>';
            } else {
                html += '<div class="messages__item messages__item--operator">' + item.message + '</div>';
            }
        });

        const chatmessage = chatbox.querySelector('.chatbox__messages');
        chatmessage.innerHTML = html;
    }

    // Chuyển sang AI Bot
    switchToAiBot() {
        this.currentBotMode = 'ai';

        // Thay đổi nội dung header
        this.args.header.innerHTML = `
            <h4 class="chatbox__heading--header">AI Chatbot</h4>
            <p class="chatbox__description--header">Chào mừng! Tôi là Suirad - AI Bot thông minh của bạn 🤖🤖 </p>
        `;
        this.args.chatBox.classList.add('chatbox--ai');
    }

    // Chuyển về Support Bot
    switchToSupportBot() {
        this.currentBotMode = 'support';

        // Thay đổi nội dung header
        this.args.header.innerHTML = `
            <h4 class="chatbox__heading--header">Chatbot Hỗ trợ</h4>
            <p class="chatbox__description--header">Xin chào! Mình là Mei - Chatbot của bạn 😇😇😇</p>
        `;
        this.args.chatBox.classList.remove('chatbox--ai');
    }
}

// Khởi tạo chatbox và hiển thị
const chatbox = new Chatbox();
chatbox.display();
