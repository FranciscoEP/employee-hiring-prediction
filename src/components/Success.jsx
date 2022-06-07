import { Box, Heading, Text } from '@chakra-ui/react';
import { CheckCircleIcon } from '@chakra-ui/icons';

export default function Success() {
    return (
        <Box textAlign="center" py={10} px={6}>
            <CheckCircleIcon boxSize="50px" color="green.500" />
            <Heading as="h2" size="xl" mt={6} mb={2}>
                Applicant could be hired
            </Heading>
            <Text color="gray.500">
                According to our model, the params given could put on track it
                for a further recognition interview.
            </Text>
        </Box>
    );
}
