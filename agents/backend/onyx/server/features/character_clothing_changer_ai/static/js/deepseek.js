/**
 * DeepSeek API Module
 * ===================
 * 
 * Handles integration with DeepSeek API for AI-powered features.
 */

const DeepSeek = {
    API_KEY: 'sk-753365753f074509bb52496e038691f6',
    BASE_URL: 'https://api.deepseek.com/v1/chat/completions',

    /**
     * Enhance prompt using DeepSeek
     * @param {string} clothingDescription - Basic clothing description
     * @param {string} characterContext - Character context (optional)
     * @returns {Promise<string>} Enhanced prompt
     */
    async enhancePrompt(clothingDescription, characterContext = '') {
        try {
            const prompt = `You are a prompt engineering expert for AI image generation. 
Enhance this clothing description for a character clothing change AI model: "${clothingDescription}"
${characterContext ? `Character context: ${characterContext}` : ''}

Return ONLY an improved, detailed prompt that:
1. Is clear and specific about the clothing
2. Includes relevant style details
3. Maintains character consistency
4. Is optimized for Flux2 model
5. Is in English

Do not include explanations, just the enhanced prompt.`;

            const response = await fetch(this.BASE_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.API_KEY}`
                },
                body: JSON.stringify({
                    model: 'deepseek-chat',
                    messages: [
                        {
                            role: 'system',
                            content: 'You are a prompt engineering expert for AI image generation models.'
                        },
                        {
                            role: 'user',
                            content: prompt
                        }
                    ],
                    temperature: 0.7,
                    max_tokens: 200
                })
            });

            if (!response.ok) {
                throw new Error(`DeepSeek API error: ${response.statusText}`);
            }

            const data = await response.json();
            return data.choices[0]?.message?.content?.trim() || clothingDescription;
        } catch (error) {
            console.warn('DeepSeek enhancement failed:', error);
            return clothingDescription; // Fallback to original
        }
    },

    /**
     * Analyze image and suggest improvements
     * @param {string} imageBase64 - Base64 encoded image
     * @returns {Promise<Object>} Analysis results
     */
    async analyzeImage(imageBase64) {
        try {
            const response = await fetch(this.BASE_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.API_KEY}`
                },
                body: JSON.stringify({
                    model: 'deepseek-chat',
                    messages: [
                        {
                            role: 'system',
                            content: 'You are an expert at analyzing character images for clothing changes. Provide brief, actionable suggestions.'
                        },
                        {
                            role: 'user',
                            content: `Analyze this character image and suggest:
1. Best areas for clothing changes
2. Recommended clothing styles
3. Any potential issues

Image data: ${imageBase64.substring(0, 100)}...`
                        }
                    ],
                    temperature: 0.5,
                    max_tokens: 300
                })
            });

            if (!response.ok) {
                throw new Error(`DeepSeek API error: ${response.statusText}`);
            }

            const data = await response.json();
            return {
                suggestions: data.choices[0]?.message?.content || '',
                success: true
            };
        } catch (error) {
            console.warn('DeepSeek analysis failed:', error);
            return {
                suggestions: '',
                success: false
            };
        }
    }
};


