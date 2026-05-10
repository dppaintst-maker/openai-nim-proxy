// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// Validate NIM_API_KEY at startup - CRITICAL
if (!NIM_API_KEY) {
  console.error('ERROR CRITICAL: NIM_API_KEY environment variable is not configured');
  console.error('Set this variable in Railway settings or in your .env file');
  process.exit(1);
}

// Toggles for reasoning display and thinking mode
const SHOW_REASONING = false;
const ENABLE_THINKING_MODE = false;

// Model mapping (adjust based on available NIM models)
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4': 'qwen/qwen3-coder-480b-a35b-instruct',
  'gpt-4-turbo': 'moonshotai/kimi-k2-instruct-0905',
  'gpt-4o': 'deepseek-ai/deepseek-v3.1',
  'claude-3-opus': 'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'qwen/qwen3-next-80b-a3b-thinking'
};

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE
  });
});

// List models endpoint (OpenAI compatible)
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));

  res.json({
    object: 'list',
    data: models
  });
});

// Chat completions endpoint (main proxy)
app.post('/v1/chat/completions', async (req, res) => {
  try {
    // Input validation - validate required fields
    const { model, messages, temperature, max_tokens, stream } = req.body;

    if (!model) {
      return res.status(400).json({
        error: {
          message: 'The "model" field is required',
          type: 'invalid_request_error',
          code: 400
        }
      });
    }

    if (!messages || !Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json({
        error: {
          message: 'The "messages" field must be a non-empty array',
          type: 'invalid_request_error',
          code: 400
        }
      });
    }

    // Smart model selection with fallback
    let nimModel = MODEL_MAPPING[model];
    if (!nimModel) {
      try {
        const testRes = await axios.post(
          NIM_API_BASE + '/chat/completions',
          {
            model: model,
            messages: [{ role: 'user', content: 'test' }],
            max_tokens: 1
          },
          {
            headers: {
              'Authorization': 'Bearer ' + NIM_API_KEY,
              'Content-Type': 'application/json'
            },
            validateStatus: function(status) { return status < 500; },
            timeout: 5000
          }
        );

        if (testRes.status >= 200 && testRes.status < 300) {
          nimModel = model;
        }
      } catch (e) {
        console.warn('Model ' + model + ' not available, using fallback');
      }

      if (!nimModel) {
        const modelLower = model.toLowerCase();
        if (modelLower.includes('gpt-4') || modelLower.includes('claude-opus') || modelLower.includes('405b')) {
          nimModel = 'meta/llama-3.1-405b-instruct';
        } else if (modelLower.includes('claude') || modelLower.includes('gemini') || modelLower.includes('70b')) {
          nimModel = 'meta/llama-3.1-70b-instruct';
        } else {
          nimModel = 'meta/llama-3.1-8b-instruct';
        }
      }
    }

    // Build NIM request - without undefined values
    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature || 0.6,
      max_tokens: max_tokens || 9024,
      stream: stream || false
    };

    // Add thinking mode only if enabled
    if (ENABLE_THINKING_MODE) {
      nimRequest.extra_body = {
        chat_template_kwargs: { thinking: true }
      };
    }

    // Make request to NVIDIA NIM API
    const response = await axios.post(NIM_API_BASE + '/chat/completions', nimRequest, {
      headers: {
        'Authorization': 'Bearer ' + NIM_API_KEY,
        'Content-Type': 'application/json'
      },
      timeout: 30000
    });

    if (stream) {
      // Streaming not fully implemented yet - return error for now
      res.json({
        error: {
          message: 'Streaming is not fully implemented. Use stream: false',
          type: 'invalid_request_error'
        }
      });
      return;
    }

    // Transform NIM response to OpenAI format
    const openaiResponse = {
      id: 'chatcmpl-' + Date.now(),
      object: 'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model: model,
      choices: response.data.choices.map(choice => {
        let fullContent = choice.message && choice.message.content ? choice.message.content : '';

        if (SHOW_REASONING && choice.message && choice.message.reasoning_content) {
          fullContent = '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' + fullContent;
        }

        return {
          index: choice.index,
          message: {
            role: choice.message.role,
            content: fullContent
          },
          finish_reason: choice.finish_reason
        };
      }),
      usage: response.data.usage || {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0
      }
    };

    res.json(openaiResponse);

  } catch (error) {
    console.error('Proxy error: ' + error.message);

    // Better error handling
    const statusCode = (error.response && error.response.status) ? error.response.status : 500;
    const errorMessage = (error.response && error.response.data && error.response.data.error && error.response.data.error.message) 
      ? error.response.data.error.message 
      : (error.message || 'Internal server error');

    res.status(statusCode).json({
      error: {
        message: errorMessage,
        type: 'invalid_request_error',
        code: statusCode
      }
    });
  }
});

// Catch-all for unsupported endpoints
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: 'Endpoint ' + req.path + ' not found',
      type: 'invalid_request_error',
      code: 404
    }
  });
});

// Start server
const server = app.listen(PORT, function() {
  console.log('OpenAI to NVIDIA NIM Proxy running on port ' + PORT);
  console.log('Health check: http://localhost:' + PORT + '/health');
  console.log('Reasoning display: ' + (SHOW_REASONING ? 'ENABLED' : 'DISABLED'));
  console.log('Thinking mode: ' + (ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'));
});

// Graceful shutdown
process.on('SIGTERM', function() {
  console.log('SIGTERM received, closing server...');
  server.close(function() {
    console.log('Server closed');
    process.exit(0);
  });
});

process.on('SIGINT', function() {
  console.log('SIGINT received, closing server...');
  server.close(function() {
    console.log('Server closed');
    process.exit(0);
  });
});
