// Frontend JavaScript
// Get DOM elements
const chatMessages = document.getElementById('chatMessages');
const messageInput = document.getElementById('messageInput');

// Session config
const STUDENT_ID = "239645";  // Chayma — demo student
const SESSION_ID = "session_" + Date.now();  // unique per browser tab

// ── Markdown rendering helpers ──────────────────────────────

function renderMarkdown(text) {
    // Use marked.js to convert markdown → HTML
    if (typeof marked !== 'undefined') {
        marked.setOptions({ breaks: true, gfm: true });
        return marked.parse(text);
    }
    // Fallback: basic manual conversion
    return text
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/- (.+)/g, '<li>$1</li>')
        .replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>')
        .replace(/\n/g, '<br>');
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Auto responses based on keywords
const responses = {
    'مرحبا': 'مرحباً بك! 😊 كيف يمكنني مساعدتك اليوم؟',
    'امتحانات': 'يمكنك الوصول إلى امتحاناتك من خلال قسم "الامتحانات" في القائمة الرئيسية. هل تحتاج مساعدة في امتحان معين؟ 📝',
    'المواد الدراسية': 'نقدم مواد دراسية تفاعلية في جميع المواد الأساسية. ما هي المادة التي تهتم بها؟ 📚',
    'نصائح الدراسة': 'إليك بعض النصائح المفيدة:\n• خصص وقتاً محدداً للدراسة يومياً\n• خذ فترات راحة منتظمة\n• استخدم التمارين التفاعلية\n• راجع المواد بانتظام 💡',
    'الدعم الفني': 'فريق الدعم الفني جاهز لمساعدتك! يمكنك التواصل معنا عبر البريد الإلكتروني: support@classquiz.com 🛠️',
    'شكرا': 'العفو! سعيد بمساعدتك. هل هناك شيء آخر تحتاج مساعدة فيه؟ 😊',
    'default': 'شكراً لرسالتك! سأحاول مساعدتك. هل يمكنك توضيح سؤالك أكثر؟ 🤔'
};

// Send message function
async function sendMessage() {
    const message = messageInput.value.trim();
    if (message === '') return;

    // Add user message
    addMessage(message, 'user');
    messageInput.value = '';

    // Show typing indicator
    showTypingIndicator();

    try {
        // Call backend API
        const response = await fetch('http://localhost:5000/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                student_id: STUDENT_ID,
                session_id: SESSION_ID
            })
        });

        if (!response.ok) throw new Error('Network response was not ok');
        
        const data = await response.json();
        hideTypingIndicator();
        addMessage(data.response, 'bot');
    } catch (error) {
        hideTypingIndicator();
        addMessage('عذراً، حدث خطأ في الاتصال بالخادم 😔', 'bot');
        console.error('Error:', error);
    }
}

// Add message to chat
function addMessage(text, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;

    const avatar = sender === 'bot' ? '🤖' : '👤';
    const time = new Date().toLocaleTimeString('ar-EG', { hour: '2-digit', minute: '2-digit' });

    // Render markdown for bot messages, plain text for user
    const rendered = sender === 'bot' ? renderMarkdown(text) : escapeHtml(text);

    messageDiv.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">
            <div class="message-bubble">${rendered}</div>
            <div class="message-time">${time}</div>
        </div>
    `;

    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Get bot response based on user message
function getBotResponse(message) {
    const lowerMessage = message.toLowerCase();
    
    for (let key in responses) {
        if (lowerMessage.includes(key)) {
            return responses[key];
        }
    }
    
    return responses['default'];
}

// Show typing indicator
function showTypingIndicator() {
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message bot';
    typingDiv.id = 'typingIndicator';
    typingDiv.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="message-content">
            <div class="message-bubble">
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        </div>
    `;
    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Hide typing indicator
function hideTypingIndicator() {
    const indicator = document.getElementById('typingIndicator');
    if (indicator) {
        indicator.remove();
    }
}

// Send quick reply
function sendQuickReply(text) {
    messageInput.value = text;
    sendMessage();
}

// Handle Enter key press
function handleKeyPress(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
}

// Insert random emoji
function insertEmoji() {
    const emojis = ['😊', '❤️', '👍', '🎉', '📚', '✨', '🌟', '💡'];
    const randomEmoji = emojis[Math.floor(Math.random() * emojis.length)];
    messageInput.value += randomEmoji;
    messageInput.focus();
}

// Toggle menu
function toggleMenu() {
    alert('القائمة قيد التطوير! 🚀');
}

// Auto-scroll to bottom on load
window.addEventListener('load', () => {
    chatMessages.scrollTop = chatMessages.scrollHeight;
});